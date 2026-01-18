from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple
import torch.nn.functional as F
from torch import nn, Tensor
from ..ops.misc import Conv2dNormActivation
from ..utils import _log_api_usage_once
def get_result_from_inner_blocks(self, x: Tensor, idx: int) -> Tensor:
    """
        This is equivalent to self.inner_blocks[idx](x),
        but torchscript doesn't support this yet
        """
    num_blocks = len(self.inner_blocks)
    if idx < 0:
        idx += num_blocks
    out = x
    for i, module in enumerate(self.inner_blocks):
        if i == idx:
            out = module(x)
    return out