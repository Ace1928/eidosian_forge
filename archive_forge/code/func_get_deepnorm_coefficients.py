from enum import Enum
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from xformers import _is_triton_available
from collections import namedtuple
def get_deepnorm_coefficients(encoder_layers: int, decoder_layers: int) -> Tuple[Optional[DeepNormCoefficients], Optional[DeepNormCoefficients]]:
    """
    See DeepNet_.

    Returns alpha and beta depending on the number of encoder and decoder layers,
    first tuple is for the encoder and second for the decoder

    .. _DeepNet: https://arxiv.org/pdf/2203.00555v1.pdf
    """
    N = encoder_layers
    M = decoder_layers
    if decoder_layers == 0:
        return (DeepNormCoefficients(alpha=(2 * N) ** 0.25, beta=(8 * N) ** (-0.25)), None)
    elif encoder_layers == 0:
        return (None, DeepNormCoefficients(alpha=(2 * M) ** 0.25, beta=(8 * M) ** (-0.25)))
    else:
        encoder_coeffs = DeepNormCoefficients(alpha=0.81 * (N ** 4 * M) ** 0.0625, beta=0.87 * (N ** 4 * M) ** (-0.0625))
        decoder_coeffs = DeepNormCoefficients(alpha=(3 * M) ** 0.25, beta=(12 * M) ** (-0.25))
        return (encoder_coeffs, decoder_coeffs)