import threading
import torch._C._lazy
from torch.utils._pytree import tree_flatten, tree_unflatten
from .closure import add_step_closure, run_step_closures
def sync_multi(tensors, devices):
    """
    Sync the list of lazy tensors so there IR get lowered for the activate backend
    and the compiled computation graph get cached.
    """
    torch._C._lazy._sync_multi(tensors, devices)