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
def mvdr_weights_rtf(rtf: Tensor, psd_n: Tensor, reference_channel: Optional[Union[int, Tensor]]=None, diagonal_loading: bool=True, diag_eps: float=1e-07, eps: float=1e-08) -> Tensor:
    """Compute the Minimum Variance Distortionless Response (*MVDR* :cite:`capon1969high`) beamforming weights
    based on the relative transfer function (RTF) and power spectral density (PSD) matrix of noise.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Given the relative transfer function (RTF) matrix or the steering vector of target speech :math:`\\bm{v}`,
    the PSD matrix of noise :math:`\\bf{\\Phi}_{\\textbf{NN}}`, and a one-hot vector that represents the
    reference channel :math:`\\bf{u}`, the method computes the MVDR beamforming weight martrix
    :math:`\\textbf{w}_{\\text{MVDR}}`. The formula is defined as:

    .. math::
        \\textbf{w}_{\\text{MVDR}}(f) =
        \\frac{{{\\bf{\\Phi}_{\\textbf{NN}}^{-1}}(f){\\bm{v}}(f)}}
        {{\\bm{v}^{\\mathsf{H}}}(f){\\bf{\\Phi}_{\\textbf{NN}}^{-1}}(f){\\bm{v}}(f)}

    where :math:`(.)^{\\mathsf{H}}` denotes the Hermitian Conjugate operation.

    Args:
        rtf (torch.Tensor): The complex-valued RTF vector of target speech.
            Tensor with dimensions `(..., freq, channel)`.
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
    if rtf.ndim < 2:
        raise ValueError(f'Expected at least 2D Tensor (..., freq, channel) for rtf. Found {rtf.shape}.')
    if psd_n.ndim < 3:
        raise ValueError(f'Expected at least 3D Tensor (..., freq, channel, channel) for psd_n. Found {psd_n.shape}.')
    if not (rtf.is_complex() and psd_n.is_complex()):
        raise TypeError(f'The type of rtf and psd_n must be ``torch.cfloat`` or ``torch.cdouble``. Found {rtf.dtype} for rtf and {psd_n.dtype} for psd_n.')
    if rtf.shape != psd_n.shape[:-1]:
        raise ValueError(f'The dimensions of rtf and the dimensions withou the last dimension of psd_n should be the same. Found {rtf.shape} for rtf and {psd_n.shape} for psd_n.')
    if psd_n.shape[-1] != psd_n.shape[-2]:
        raise ValueError(f'The last two dimensions of psd_n should be the same. Found {psd_n.shape}.')
    if diagonal_loading:
        psd_n = _tik_reg(psd_n, reg=diag_eps)
    numerator = torch.linalg.solve(psd_n, rtf.unsqueeze(-1)).squeeze(-1)
    denominator = torch.einsum('...d,...d->...', [rtf.conj(), numerator])
    beamform_weights = numerator / (denominator.real.unsqueeze(-1) + eps)
    if reference_channel is not None:
        if torch.jit.isinstance(reference_channel, int):
            scale = rtf[..., reference_channel].conj()
        elif torch.jit.isinstance(reference_channel, Tensor):
            reference_channel = reference_channel.to(psd_n.dtype)
            scale = torch.einsum('...c,...c->...', [rtf.conj(), reference_channel[..., None, :]])
        else:
            raise TypeError(f'Expected "int" or "Tensor" for reference_channel. Found: {type(reference_channel)}.')
        beamform_weights = beamform_weights * scale[..., None]
    return beamform_weights