import warnings
from typing import List, Optional, Union
import torch
from torchaudio.functional import fftconvolve
def extend_pitch(base: torch.Tensor, pattern: Union[int, List[float], torch.Tensor]):
    """Extend the given time series values with multipliers of them.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Given a series of fundamental frequencies (pitch), this function appends
    its harmonic overtones or inharmonic partials.

    Args:
        base (torch.Tensor):
            Base time series, like fundamental frequencies (Hz). Shape: `(..., time, 1)`.
        pattern (int, list of floats or torch.Tensor):
            If ``int``, the number of pitch series after the operation.
            `pattern - 1` tones are added, so that the resulting Tensor contains
            up to `pattern`-th overtones of the given series.

            If list of float or ``torch.Tensor``, it must be one dimensional,
            representing the custom multiplier of the fundamental frequency.

    Returns:
        Tensor: Oscillator frequencies (Hz). Shape: `(..., time, num_tones)`.

    Example
        >>> # fundamental frequency
        >>> f0 = torch.linspace(1, 5, 5).unsqueeze(-1)
        >>> f0
        tensor([[1.],
                [2.],
                [3.],
                [4.],
                [5.]])
        >>> # Add harmonic overtones, up to 3rd.
        >>> f = extend_pitch(f0, 3)
        >>> f.shape
        torch.Size([5, 3])
        >>> f
        tensor([[ 1.,  2.,  3.],
                [ 2.,  4.,  6.],
                [ 3.,  6.,  9.],
                [ 4.,  8., 12.],
                [ 5., 10., 15.]])
        >>> # Add custom (inharmonic) partials.
        >>> f = extend_pitch(f0, torch.tensor([1, 2.1, 3.3, 4.5]))
        >>> f.shape
        torch.Size([5, 4])
        >>> f
        tensor([[ 1.0000,  2.1000,  3.3000,  4.5000],
                [ 2.0000,  4.2000,  6.6000,  9.0000],
                [ 3.0000,  6.3000,  9.9000, 13.5000],
                [ 4.0000,  8.4000, 13.2000, 18.0000],
                [ 5.0000, 10.5000, 16.5000, 22.5000]])
    """
    if isinstance(pattern, torch.Tensor):
        mult = pattern
    elif isinstance(pattern, int):
        mult = torch.linspace(1.0, float(pattern), pattern, device=base.device, dtype=base.dtype)
    else:
        mult = torch.tensor(pattern, dtype=base.dtype, device=base.device)
    h_freq = base @ mult.unsqueeze(0)
    return h_freq