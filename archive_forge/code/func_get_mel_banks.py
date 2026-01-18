import math
from typing import Tuple
import torch
import torchaudio
from torch import Tensor
def get_mel_banks(num_bins: int, window_length_padded: int, sample_freq: float, low_freq: float, high_freq: float, vtln_low: float, vtln_high: float, vtln_warp_factor: float) -> Tuple[Tensor, Tensor]:
    """
    Returns:
        (Tensor, Tensor): The tuple consists of ``bins`` (which is
        melbank of size (``num_bins``, ``num_fft_bins``)) and ``center_freqs`` (which is
        center frequencies of bins of size (``num_bins``)).
    """
    assert num_bins > 3, 'Must have at least 3 mel bins'
    assert window_length_padded % 2 == 0
    num_fft_bins = window_length_padded / 2
    nyquist = 0.5 * sample_freq
    if high_freq <= 0.0:
        high_freq += nyquist
    assert 0.0 <= low_freq < nyquist and 0.0 < high_freq <= nyquist and (low_freq < high_freq), 'Bad values in options: low-freq {} and high-freq {} vs. nyquist {}'.format(low_freq, high_freq, nyquist)
    fft_bin_width = sample_freq / window_length_padded
    mel_low_freq = mel_scale_scalar(low_freq)
    mel_high_freq = mel_scale_scalar(high_freq)
    mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins + 1)
    if vtln_high < 0.0:
        vtln_high += nyquist
    assert vtln_warp_factor == 1.0 or (low_freq < vtln_low < high_freq and 0.0 < vtln_high < high_freq and (vtln_low < vtln_high)), 'Bad values in options: vtln-low {} and vtln-high {}, versus low-freq {} and high-freq {}'.format(vtln_low, vtln_high, low_freq, high_freq)
    bin = torch.arange(num_bins).unsqueeze(1)
    left_mel = mel_low_freq + bin * mel_freq_delta
    center_mel = mel_low_freq + (bin + 1.0) * mel_freq_delta
    right_mel = mel_low_freq + (bin + 2.0) * mel_freq_delta
    if vtln_warp_factor != 1.0:
        left_mel = vtln_warp_mel_freq(vtln_low, vtln_high, low_freq, high_freq, vtln_warp_factor, left_mel)
        center_mel = vtln_warp_mel_freq(vtln_low, vtln_high, low_freq, high_freq, vtln_warp_factor, center_mel)
        right_mel = vtln_warp_mel_freq(vtln_low, vtln_high, low_freq, high_freq, vtln_warp_factor, right_mel)
    center_freqs = inverse_mel_scale(center_mel)
    mel = mel_scale(fft_bin_width * torch.arange(num_fft_bins)).unsqueeze(0)
    up_slope = (mel - left_mel) / (center_mel - left_mel)
    down_slope = (right_mel - mel) / (right_mel - center_mel)
    if vtln_warp_factor == 1.0:
        bins = torch.max(torch.zeros(1), torch.min(up_slope, down_slope))
    else:
        bins = torch.zeros_like(up_slope)
        up_idx = torch.gt(mel, left_mel) & torch.le(mel, center_mel)
        down_idx = torch.gt(mel, center_mel) & torch.lt(mel, right_mel)
        bins[up_idx] = up_slope[up_idx]
        bins[down_idx] = down_slope[down_idx]
    return (bins, center_freqs)