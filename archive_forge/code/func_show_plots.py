from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from functools import lru_cache
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Sequence, Set, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle
from fairscale.nn import FullyShardedDataParallel
def show_plots(self, figsize: Tuple[int, int]=(16, 20), capture: bool=False) -> Optional[Any]:
    """
        Show useful memory plots. Use "capture=True" to return an image
        rather than displaying the plots.
        """
    return compare_memory_traces_in_plot({'run': self.memory_traces}, figsize=figsize, capture=capture)