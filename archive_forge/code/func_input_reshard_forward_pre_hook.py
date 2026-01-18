from functools import partial
from typing import Any, Optional, Tuple
import torch
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard
def input_reshard_forward_pre_hook(_: torch.nn.Module, _i: Tuple[Any, ...]) -> None:
    saved_tensor_hooks = torch.autograd.graph.saved_tensors_hooks(partial(_pack_hook_tp, tp_device_mesh, input_reshard_dim), partial(_unpack_hook_tp, tp_device_mesh, input_reshard_dim))
    saved_tensor_hooks.__enter__()
    nonlocal cx
    cx = saved_tensor_hooks