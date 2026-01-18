from functools import lru_cache
from math import ceil, pi
from typing import Optional, Tuple
import torch
from torch import Tensor
from torch.nn.functional import pad
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.imports import (
def speech_reverberation_modulation_energy_ratio(preds: Tensor, fs: int, n_cochlear_filters: int=23, low_freq: float=125, min_cf: float=4, max_cf: Optional[float]=None, norm: bool=False, fast: bool=False) -> Tensor:
    """Calculate `Speech-to-Reverberation Modulation Energy Ratio`_ (SRMR).

    SRMR is a non-intrusive metric for speech quality and intelligibility based on
    a modulation spectral representation of the speech signal.
    This code is translated from `SRMRToolbox`_ and `SRMRpy`_.

    Args:
        preds: shape ``(..., time)``
        fs: the sampling rate
        n_cochlear_filters: Number of filters in the acoustic filterbank
        low_freq: determines the frequency cutoff for the corresponding gammatone filterbank.
        min_cf: Center frequency in Hz of the first modulation filter.
        max_cf: Center frequency in Hz of the last modulation filter. If None is given,
            then 30 Hz will be used for `norm==False`, otherwise 128 Hz will be used.
        norm: Use modulation spectrum energy normalization
        fast: Use the faster version based on the gammatonegram.
            Note: this argument is inherited from `SRMRpy`_. As the translated code is based to pytorch,
            setting `fast=True` may slow down the speed for calculating this metric on GPU.

    .. note:: using this metrics requires you to have ``gammatone`` and ``torchaudio`` installed.
        Either install as ``pip install torchmetrics[audio]`` or ``pip install torchaudio``
        and ``pip install git+https://github.com/detly/gammatone``.

    .. note::
        This implementation is experimental, and might not be consistent with the matlab
        implementation `SRMRToolbox`_, especially the fast implementation.
        The slow versions, a) fast=False, norm=False, max_cf=128, b) fast=False, norm=True, max_cf=30, have
        a relatively small inconsistence.

    Returns:
        Scalar tensor with srmr value with shape ``(...)``

    Raises:
        ModuleNotFoundError:
            If ``gammatone`` or ``torchaudio`` package is not installed

    Example:
        >>> import torch
        >>> from torchmetrics.functional.audio import speech_reverberation_modulation_energy_ratio
        >>> g = torch.manual_seed(1)
        >>> preds = torch.randn(8000)
        >>> speech_reverberation_modulation_energy_ratio(preds, 8000)
        tensor([0.3354], dtype=torch.float64)

    """
    if not _TORCHAUDIO_AVAILABLE or not _TORCHAUDIO_GREATER_EQUAL_0_10 or (not _GAMMATONE_AVAILABLE):
        raise ModuleNotFoundError('speech_reverberation_modulation_energy_ratio requires you to have `gammatone` and `torchaudio>=0.10` installed. Either install as ``pip install torchmetrics[audio]`` or ``pip install torchaudio>=0.10`` and ``pip install git+https://github.com/detly/gammatone``')
    from gammatone.fftweight import fft_gtgram
    from torchaudio.functional.filtering import lfilter
    _srmr_arg_validate(fs=fs, n_cochlear_filters=n_cochlear_filters, low_freq=low_freq, min_cf=min_cf, max_cf=max_cf, norm=norm, fast=fast)
    shape = preds.shape
    preds = preds.reshape(1, -1) if len(shape) == 1 else preds.reshape(-1, shape[-1])
    num_batch, time = preds.shape
    if not torch.is_floating_point(preds):
        preds = preds.to(torch.float64) / torch.finfo(preds.dtype).max
    max_vals = preds.abs().max(dim=-1, keepdim=True).values
    val_norm = torch.where(max_vals > 1, max_vals, torch.tensor(1.0, dtype=max_vals.dtype, device=max_vals.device))
    preds = preds / val_norm
    w_length_s = 0.256
    w_inc_s = 0.064
    if fast:
        rank_zero_warn('`fast=True` may slow down the speed of SRMR metric on GPU.')
        mfs = 400.0
        temp = []
        preds_np = preds.detach().cpu().numpy()
        for b in range(num_batch):
            gt_env_b = fft_gtgram(preds_np[b], fs, 0.01, 0.0025, n_cochlear_filters, low_freq)
            temp.append(torch.tensor(gt_env_b))
        gt_env = torch.stack(temp, dim=0).to(device=preds.device)
    else:
        fcoefs = _make_erb_filters(fs, n_cochlear_filters, low_freq, device=preds.device)
        gt_env = torch.abs(_hilbert(_erb_filterbank(preds, fcoefs)))
        mfs = fs
    w_length = ceil(w_length_s * mfs)
    w_inc = ceil(w_inc_s * mfs)
    if max_cf is None:
        max_cf = 30 if norm else 128
    _, mf, cutoffs, _ = _compute_modulation_filterbank_and_cutoffs(min_cf, max_cf, n=8, fs=mfs, q=2, device=preds.device)
    num_frames = int(1 + (time - w_length) // w_inc)
    w = torch.hamming_window(w_length + 1, dtype=torch.float64, device=preds.device)[:-1]
    mod_out = lfilter(gt_env.unsqueeze(-2).expand(-1, -1, mf.shape[0], -1), mf[:, 1, :], mf[:, 0, :], clamp=False, batching=True)
    padding = (0, max(ceil(time / w_inc) * w_inc - time, w_length - time))
    mod_out_pad = pad(mod_out, pad=padding, mode='constant', value=0)
    mod_out_frame = mod_out_pad.unfold(-1, w_length, w_inc)
    energy = ((mod_out_frame[..., :num_frames, :] * w) ** 2).sum(dim=-1)
    if norm:
        energy = _normalize_energy(energy)
    erbs = torch.flipud(_calc_erbs(low_freq, fs, n_cochlear_filters, device=preds.device))
    avg_energy = torch.mean(energy, dim=-1)
    total_energy = torch.sum(avg_energy.reshape(num_batch, -1), dim=-1)
    ac_energy = torch.sum(avg_energy, dim=2)
    ac_perc = ac_energy * 100 / total_energy.reshape(-1, 1)
    ac_perc_cumsum = ac_perc.flip(-1).cumsum(-1)
    k90perc_idx = torch.nonzero((ac_perc_cumsum > 90).cumsum(-1) == 1)[:, 1]
    bw = erbs[k90perc_idx]
    temp = []
    for b in range(num_batch):
        score = _cal_srmr_score(bw[b], avg_energy[b], cutoffs=cutoffs)
        temp.append(score)
    score = torch.stack(temp)
    return score.reshape(*shape[:-1]) if len(shape) > 1 else score