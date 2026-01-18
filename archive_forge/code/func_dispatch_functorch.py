from abc import ABC, abstractmethod
import contextlib
from typing import Any
import torch
import torch.utils._pytree as pytree
from torch._C._functorch import (
from torch.autograd.forward_ad import _set_fwd_grad_enabled
def dispatch_functorch(op, args, kwargs):
    interpreter = retrieve_current_functorch_interpreter()
    args, kwargs = pytree.tree_map_only(torch.Tensor, torch._C._functorch.unwrap_if_dead, (args, kwargs))
    return interpreter.process(op, args, kwargs)