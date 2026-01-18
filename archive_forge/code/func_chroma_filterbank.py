import math
import warnings
from typing import Optional
import torch
from torchaudio.functional.functional import _create_triangular_filterbank
def chroma_filterbank(sample_rate: int, n_freqs: int, n_chroma: int, *, tuning: float=0.0, ctroct: float=5.0, octwidth: Optional[float]=2.0, norm: int=2, base_c: bool=True):
    """Create a frequency-to-chroma conversion matrix. Implementation adapted from librosa.

    Args:
        sample_rate (int): Sample rate.
        n_freqs (int): Number of input frequencies.
        n_chroma (int): Number of output chroma.
        tuning (float, optional): Tuning deviation from A440 in fractions of a chroma bin. (Default: 0.0)
        ctroct (float, optional): Center of Gaussian dominance window to weight filters by, in octaves. (Default: 5.0)
        octwidth (float or None, optional): Width of Gaussian dominance window to weight filters by, in octaves.
            If ``None``, then disable weighting altogether. (Default: 2.0)
        norm (int, optional): order of norm to normalize filter bank by. (Default: 2)
        base_c (bool, optional): If True, then start filter bank at C. Otherwise, start at A. (Default: True)

    Returns:
        torch.Tensor: Chroma filter bank, with shape `(n_freqs, n_chroma)`.
    """
    freqs = torch.linspace(0, sample_rate // 2, n_freqs)[1:]
    freq_bins = n_chroma * _hz_to_octs(freqs, bins_per_octave=n_chroma, tuning=tuning)
    freq_bins = torch.cat((torch.tensor([freq_bins[0] - 1.5 * n_chroma]), freq_bins))
    freq_bin_widths = torch.cat((torch.maximum(freq_bins[1:] - freq_bins[:-1], torch.tensor(1.0)), torch.tensor([1])))
    D = freq_bins.unsqueeze(1) - torch.arange(0, n_chroma)
    n_chroma2 = round(n_chroma / 2)
    D = torch.remainder(D + n_chroma2, n_chroma) - n_chroma2
    fb = torch.exp(-0.5 * (2 * D / torch.tile(freq_bin_widths.unsqueeze(1), (1, n_chroma))) ** 2)
    fb = torch.nn.functional.normalize(fb, p=norm, dim=1)
    if octwidth is not None:
        fb *= torch.tile(torch.exp(-0.5 * ((freq_bins.unsqueeze(1) / n_chroma - ctroct) / octwidth) ** 2), (1, n_chroma))
    if base_c:
        fb = torch.roll(fb, -3 * (n_chroma // 12), dims=1)
    return fb