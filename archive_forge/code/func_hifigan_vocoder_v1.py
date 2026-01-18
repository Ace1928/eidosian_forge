from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
def hifigan_vocoder_v1() -> HiFiGANVocoder:
    """Builds HiFiGAN Vocoder with V1 architecture :cite:`NEURIPS2020_c5d73680`.

    Returns:
        HiFiGANVocoder: generated model.
    """
    return hifigan_vocoder(upsample_rates=(8, 8, 2, 2), upsample_kernel_sizes=(16, 16, 4, 4), upsample_initial_channel=512, resblock_kernel_sizes=(3, 7, 11), resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)), resblock_type=1, in_channels=80, lrelu_slope=0.1)