import itertools
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Set, Tuple
import torch.cuda.memory
import torch.cuda.nvtx
import torch.profiler
import torch.utils.hooks
from torch.utils._python_dispatch import TorchDispatchMode, _pop_mode_temporarily
from torch.utils._pytree import tree_map
from ..ops.common import FUNC_TO_XFORMERS_OPERATOR
from .device_limits import get_device_limits
from .profiler import _Profiler
def conv_flop(inputs: List[Any], outputs: List[Any]):
    """
    Count flops for convolution.
    """
    x, w = inputs[:2]
    x_shape, w_shape, out_shape = (get_shape(x), get_shape(w), get_shape(outputs[0]))
    transposed = inputs[6]
    return conv_flop_count(x_shape, w_shape, out_shape, transposed=transposed)