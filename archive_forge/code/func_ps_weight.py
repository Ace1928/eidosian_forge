from enum import Enum
import logging
from typing import Iterator, List, Optional, Tuple, cast
import torch
import torch.distributed as dist
from .graph_manager import GraphManager
from .mixing_manager import MixingManager, UniformMixing
@ps_weight.setter
def ps_weight(self, v: torch.Tensor) -> None:
    self._ps_weight.data[0] = v