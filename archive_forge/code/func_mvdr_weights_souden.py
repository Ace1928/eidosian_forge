import math
import tempfile
import warnings
from collections.abc import Sequence
from typing import List, Optional, Tuple, Union
import torch
import torchaudio
from torch import Tensor
from torchaudio._internal.module_utils import deprecated
from .filtering import highpass_biquad, treble_biquad
def mvdr_weights_souden(psd_s: Tensor, psd_n: Tensor, reference_channel: Union[int, Tensor], diagonal_loading: bool=True, diag_eps: float=1e-07, eps: float=1e-08) -> Tensor:
    """Compute the Minimum Variance Distortionless Response (*MVDR* :cite:`capon1969high`) beamforming weights
    by the method proposed by *Souden et, al.* :cite:`souden2009optimal`.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Given the power spectral density (PSD) matrix of target speech :math:`\\bf{\\Phi}_{\\textbf{SS}}`,
    the PSD matrix of noise :math:`\\bf{\\Phi}_{\\textbf{NN}}`, and a one-hot vector that represents the
    reference channel :math:`\\bf{u}`, the method computes the MVDR beamforming weight martrix
    :math:`\\textbf{w}_{\\text{MVDR}}`. The formula is defined as:

    .. math::
        \\textbf{w}_{\\text{MVDR}}(f) =
        \\frac{{{\\bf{\\Phi}_{\\textbf{NN}}^{-1}}(f){\\bf{\\Phi}_{\\textbf{SS}}}}(f)}
        {\\text{Trace}({{{\\bf{\\Phi}_{\\textbf{NN}}^{-1}}(f) \\bf{\\Phi}_{\\textbf{SS}}}(f))}}\\bm{u}

    Args:
        psd_s (torch.Tensor): The complex-valued power spectral density (PSD) matrix of target speech.
            Tensor with dimensions `(..., freq, channel, channel)`.
        psd_n (torch.Tensor): The complex-valued power spectral density (PSD) matrix of noise.
            Tensor with dimensions `(..., freq, channel, channel)`.
        reference_channel (int or torch.Tensor): Specifies the reference channel.
            If the dtype is ``int``, it represents the reference channel index.
            If the dtype is ``torch.Tensor``, its shape is `(..., channel)`, where the ``channel`` dimension
            is one-hot.
        diagonal_loading (bool, optional): If ``True``, enables applying diagonal loading to ``psd_n``.
            (Default: ``True``)
        diag_eps (float, optional): The coefficient multiplied to the identity matrix for diagonal loading.
            It is only effective when ``diagonal_loading`` is set to ``True``. (Default: ``1e-7``)
        eps (float, optional): Value to add to the denominator in the beamforming weight formula.
            (Default: ``1e-8``)

    Returns:
        torch.Tensor: The complex-valued MVDR beamforming weight matrix with dimensions `(..., freq, channel)`.
    """
    _assert_psd_matrices(psd_s, psd_n)
    if diagonal_loading:
        psd_n = _tik_reg(psd_n, reg=diag_eps)
    numerator = torch.linalg.solve(psd_n, psd_s)
    ws = numerator / (_compute_mat_trace(numerator)[..., None, None] + eps)
    if torch.jit.isinstance(reference_channel, int):
        beamform_weights = ws[..., :, reference_channel]
    elif torch.jit.isinstance(reference_channel, Tensor):
        reference_channel = reference_channel.to(psd_n.dtype)
        beamform_weights = torch.einsum('...c,...c->...', [ws, reference_channel[..., None, None, :]])
    else:
        raise TypeError(f'Expected "int" or "Tensor" for reference_channel. Found: {type(reference_channel)}.')
    return beamform_weights