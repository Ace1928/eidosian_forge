import copy
import time
from typing import Any, Generator, List, Union, Sequence
import torch
from torch import Tensor
import torch.nn as nn
from ..microbatch import Batch
def layerwise_sandbox(module: nn.Sequential, device: torch.device) -> Generator[nn.Module, None, None]:
    """Copies layers for ease to profile. It doesn't modify the given
    module.
    """
    for layer in module:
        layer_copy = copy.deepcopy(layer)
        layer_copy.to(device)
        layer_copy.train()
        yield layer_copy