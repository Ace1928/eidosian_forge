import os
from typing import List, Optional, Tuple
import torch
import torchaudio
from torchaudio._internal.module_utils import deprecated
from torchaudio.utils.sox_utils import list_effects
def effect_names() -> List[str]:
    """Gets list of valid sox effect names

    Returns:
        List[str]: list of available effect names.

    Example
        >>> torchaudio.sox_effects.effect_names()
        ['allpass', 'band', 'bandpass', ... ]
    """
    return list(list_effects().keys())