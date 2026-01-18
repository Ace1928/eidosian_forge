import logging
import typing
from collections import Counter
from typing import Dict, Set
import torch
import torch._guards
from torch._inductor.constant_folding import ConstantFolder
from torch.multiprocessing.reductions import StorageWeakRef
from .. import config
from ..pattern_matcher import (
from .replace_random import replace_random_passes
def add_node_replacement(self, node: torch.fx.Node, tensor: torch.Tensor) -> None:
    self.node_replacements[node] = tensor.flatten()[0].item()
    self.constant_data_ptrs[node] = StorageWeakRef(tensor.untyped_storage())