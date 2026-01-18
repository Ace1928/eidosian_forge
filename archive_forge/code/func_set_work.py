from dataclasses import dataclass
from functools import partial
from typing import Any, List, Optional, Tuple
import torch
from torch._C import _disabled_torch_function_impl
from torch.fx.experimental.proxy_tensor import (
from torch.utils import _pytree as pytree
from torch.utils._mode_utils import no_dispatch
from torch.utils._pytree import tree_flatten, tree_map, tree_map_only
def set_work(work: torch.distributed._Work, e: Any):
    if isinstance(e, CommTensor):
        e._work = work
    elif isinstance(e, torch.Tensor):
        raise RuntimeError('Type of output tensors from collective communication during tracing should always be CommTensor instead of torch.Tensor')
    return e