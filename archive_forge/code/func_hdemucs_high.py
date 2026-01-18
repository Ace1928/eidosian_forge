import math
import typing as tp
from typing import Any, Dict, List, Optional
import torch
from torch import nn
from torch.nn import functional as F
def hdemucs_high(sources: List[str]) -> HDemucs:
    """Builds medium nfft (4096) version of :class:`HDemucs`, suitable for sample rates of 44.1-48 kHz.

    Args:
        sources (List[str]): See :py:func:`HDemucs`.

    Returns:
        HDemucs:
            HDemucs model.
    """
    return HDemucs(sources=sources, nfft=4096, depth=6)