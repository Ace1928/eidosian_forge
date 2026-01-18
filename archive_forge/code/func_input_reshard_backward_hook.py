from functools import partial
from typing import Any, Optional, Tuple
import torch
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard
def input_reshard_backward_hook(_: torch.nn.Module, _i: Tuple[Any, ...], _o: Any) -> Any:
    nonlocal cx
    cx.__exit__()