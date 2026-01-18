import contextlib
import logging
import math
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.hooks import RemovableHandle
import pytorch_lightning as pl
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
from pytorch_lightning.utilities.model_helpers import _ModuleMode
from pytorch_lightning.utilities.rank_zero import WarningCache
def get_human_readable_count(number: int) -> str:
    """Abbreviates an integer number with K, M, B, T for thousands, millions, billions and trillions, respectively.

    Examples:
        >>> get_human_readable_count(123)
        '123  '
        >>> get_human_readable_count(1234)  # (one thousand)
        '1.2 K'
        >>> get_human_readable_count(2e6)   # (two million)
        '2.0 M'
        >>> get_human_readable_count(3e9)   # (three billion)
        '3.0 B'
        >>> get_human_readable_count(4e14)  # (four hundred trillion)
        '400 T'
        >>> get_human_readable_count(5e15)  # (more than trillion)
        '5,000 T'

    Args:
        number: a positive integer number

    Return:
        A string formatted according to the pattern described above.

    """
    assert number >= 0
    labels = PARAMETER_NUM_UNITS
    num_digits = int(math.floor(math.log10(number)) + 1 if number > 0 else 1)
    num_groups = int(math.ceil(num_digits / 3))
    num_groups = min(num_groups, len(labels))
    shift = -3 * (num_groups - 1)
    number = number * 10 ** shift
    index = num_groups - 1
    if index < 1 or number >= 100:
        return f'{int(number):,d} {labels[index]}'
    return f'{number:,.1f} {labels[index]}'