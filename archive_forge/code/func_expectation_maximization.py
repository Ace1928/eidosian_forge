from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
def expectation_maximization(y: torch.Tensor, x: torch.Tensor, iterations: int=2, eps: float=1e-10, batch_size: int=200):
    """Expectation maximization algorithm, for refining source separation
    estimates.

    This algorithm allows to make source separation results better by
    enforcing multichannel consistency for the estimates. This usually means
    a better perceptual quality in terms of spatial artifacts.

    The implementation follows the details presented in [1]_, taking
    inspiration from the original EM algorithm proposed in [2]_ and its
    weighted refinement proposed in [3]_, [4]_.
    It works by iteratively:

     * Re-estimate source parameters (power spectral densities and spatial
       covariance matrices) through :func:`get_local_gaussian_model`.

     * Separate again the mixture with the new parameters by first computing
       the new modelled mixture covariance matrices with :func:`get_mix_model`,
       prepare the Wiener filters through :func:`wiener_gain` and apply them
       with :func:`apply_filter``.

    References
    ----------
    .. [1] S. Uhlich and M. Porcu and F. Giron and M. Enenkl and T. Kemp and
        N. Takahashi and Y. Mitsufuji, "Improving music source separation based
        on deep neural networks through data augmentation and network
        blending." 2017 IEEE International Conference on Acoustics, Speech
        and Signal Processing (ICASSP). IEEE, 2017.

    .. [2] N.Q. Duong and E. Vincent and R.Gribonval. "Under-determined
        reverberant audio source separation using a full-rank spatial
        covariance model." IEEE Transactions on Audio, Speech, and Language
        Processing 18.7 (2010): 1830-1840.

    .. [3] A. Nugraha and A. Liutkus and E. Vincent. "Multichannel audio source
        separation with deep neural networks." IEEE/ACM Transactions on Audio,
        Speech, and Language Processing 24.9 (2016): 1652-1664.

    .. [4] A. Nugraha and A. Liutkus and E. Vincent. "Multichannel music
        separation with deep neural networks." 2016 24th European Signal
        Processing Conference (EUSIPCO). IEEE, 2016.

    .. [5] A. Liutkus and R. Badeau and G. Richard "Kernel additive models for
        source separation." IEEE Transactions on Signal Processing
        62.16 (2014): 4298-4310.

    Args:
        y (Tensor): [shape=(nb_frames, nb_bins, nb_channels, 2, nb_sources)]
            initial estimates for the sources
        x (Tensor): [shape=(nb_frames, nb_bins, nb_channels, 2)]
            complex STFT of the mixture signal
        iterations (int): [scalar]
            number of iterations for the EM algorithm.
        eps (float or None): [scalar]
            The epsilon value to use for regularization and filters.

    Returns:
        y (Tensor): [shape=(nb_frames, nb_bins, nb_channels, 2, nb_sources)]
            estimated sources after iterations
        v (Tensor): [shape=(nb_frames, nb_bins, nb_sources)]
            estimated power spectral densities
        R (Tensor): [shape=(nb_bins, nb_channels, nb_channels, 2, nb_sources)]
            estimated spatial covariance matrices

    Notes:
        * You need an initial estimate for the sources to apply this
          algorithm. This is precisely what the :func:`wiener` function does.
        * This algorithm *is not* an implementation of the "exact" EM
          proposed in [1]_. In particular, it does compute the posterior
          covariance matrices the same (exact) way. Instead, it uses the
          simplified approximate scheme initially proposed in [5]_ and further
          refined in [3]_, [4]_, that boils down to just take the empirical
          covariance of the recent source estimates, followed by a weighted
          average for the update of the spatial covariance matrix. It has been
          empirically demonstrated that this simplified algorithm is more
          robust for music separation.

    Warning:
        It is *very* important to make sure `x.dtype` is `torch.float64`
        if you want double precision, because this function will **not**
        do such conversion for you from `torch.complex32`, in case you want the
        smaller RAM usage on purpose.

        It is usually always better in terms of quality to have double
        precision, by e.g. calling :func:`expectation_maximization`
        with ``x.to(torch.float64)``.
    """
    nb_frames, nb_bins, nb_channels = x.shape[:-1]
    nb_sources = y.shape[-1]
    regularization = torch.cat((torch.eye(nb_channels, dtype=x.dtype, device=x.device)[..., None], torch.zeros((nb_channels, nb_channels, 1), dtype=x.dtype, device=x.device)), dim=2)
    regularization = torch.sqrt(torch.as_tensor(eps)) * regularization[None, None, ...].expand((-1, nb_bins, -1, -1, -1))
    R = [torch.zeros((nb_bins, nb_channels, nb_channels, 2), dtype=x.dtype, device=x.device) for j in range(nb_sources)]
    weight: torch.Tensor = torch.zeros((nb_bins,), dtype=x.dtype, device=x.device)
    v: torch.Tensor = torch.zeros((nb_frames, nb_bins, nb_sources), dtype=x.dtype, device=x.device)
    for it in range(iterations):
        v = torch.mean(torch.abs(y[..., 0, :]) ** 2 + torch.abs(y[..., 1, :]) ** 2, dim=-2)
        for j in range(nb_sources):
            R[j] = torch.tensor(0.0, device=x.device)
            weight = torch.tensor(eps, device=x.device)
            pos: int = 0
            batch_size = batch_size if batch_size else nb_frames
            while pos < nb_frames:
                t = torch.arange(pos, min(nb_frames, pos + batch_size))
                pos = int(t[-1]) + 1
                R[j] = R[j] + torch.sum(_covariance(y[t, ..., j]), dim=0)
                weight = weight + torch.sum(v[t, ..., j], dim=0)
            R[j] = R[j] / weight[..., None, None, None]
            weight = torch.zeros_like(weight)
        if y.requires_grad:
            y = y.clone()
        pos = 0
        while pos < nb_frames:
            t = torch.arange(pos, min(nb_frames, pos + batch_size))
            pos = int(t[-1]) + 1
            y[t, ...] = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            Cxx = regularization
            for j in range(nb_sources):
                Cxx = Cxx + v[t, ..., j, None, None, None] * R[j][None, ...].clone()
            inv_Cxx = _invert(Cxx)
            for j in range(nb_sources):
                gain = torch.zeros_like(inv_Cxx)
                indices = torch.cartesian_prod(torch.arange(nb_channels), torch.arange(nb_channels), torch.arange(nb_channels))
                for index in indices:
                    gain[:, :, index[0], index[1], :] = _mul_add(R[j][None, :, index[0], index[2], :].clone(), inv_Cxx[:, :, index[2], index[1], :], gain[:, :, index[0], index[1], :])
                gain = gain * v[t, ..., None, None, None, j]
                for i in range(nb_channels):
                    y[t, ..., j] = _mul_add(gain[..., i, :], x[t, ..., i, None, :], y[t, ..., j])
    return (y, v, R)