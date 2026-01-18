import math
import typing as tp
from typing import Any, Dict, List, Optional
import torch
from torch import nn
from torch.nn import functional as F
def hdemucs_medium(sources: List[str]) -> HDemucs:
    """Builds medium nfft (2048) version of :class:`HDemucs`, suitable for sample rates of 16-32 kHz.

    .. note::

        Medium HDemucs has not been tested against the original Hybrid Demucs as this nfft and depth configuration is
        not compatible with the original implementation in https://github.com/facebookresearch/demucs

    Args:
        sources (List[str]): See :py:func:`HDemucs`.

    Returns:
        HDemucs:
            HDemucs model.
    """
    return HDemucs(sources=sources, nfft=2048, depth=6)