import operator
import torch
from torch.ao.quantization.backend_config import (
from typing import List
def root_node_getter(node_pattern):
    getitem, maxpool, index = node_pattern
    return maxpool