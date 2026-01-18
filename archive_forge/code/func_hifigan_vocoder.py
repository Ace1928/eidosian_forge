from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
def hifigan_vocoder(in_channels: int, upsample_rates: Tuple[int, ...], upsample_initial_channel: int, upsample_kernel_sizes: Tuple[int, ...], resblock_kernel_sizes: Tuple[int, ...], resblock_dilation_sizes: Tuple[Tuple[int, ...], ...], resblock_type: int, lrelu_slope: float) -> HiFiGANVocoder:
    """Builds HiFi GAN Vocoder :cite:`NEURIPS2020_c5d73680`.

    Args:
        in_channels (int): See :py:class:`HiFiGANVocoder`.
        upsample_rates (tuple of ``int``): See :py:class:`HiFiGANVocoder`.
        upsample_initial_channel (int): See :py:class:`HiFiGANVocoder`.
        upsample_kernel_sizes (tuple of ``int``): See :py:class:`HiFiGANVocoder`.
        resblock_kernel_sizes (tuple of ``int``): See :py:class:`HiFiGANVocoder`.
        resblock_dilation_sizes (tuple of tuples of ``int``): See :py:class:`HiFiGANVocoder`.
        resblock_type (int, 1 or 2): See :py:class:`HiFiGANVocoder`.
    Returns:
        HiFiGANVocoder: generated model.
    """
    return HiFiGANVocoder(upsample_rates=upsample_rates, resblock_kernel_sizes=resblock_kernel_sizes, resblock_dilation_sizes=resblock_dilation_sizes, resblock_type=resblock_type, upsample_initial_channel=upsample_initial_channel, upsample_kernel_sizes=upsample_kernel_sizes, in_channels=in_channels, lrelu_slope=lrelu_slope)