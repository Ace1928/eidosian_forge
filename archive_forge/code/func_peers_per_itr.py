from enum import Enum
import logging
from typing import Iterator, List, Optional, Tuple, cast
import torch
import torch.distributed as dist
from .graph_manager import GraphManager
from .mixing_manager import MixingManager, UniformMixing
@peers_per_itr.setter
def peers_per_itr(self, v: int) -> None:
    self._graph_manager.peers_per_itr = v