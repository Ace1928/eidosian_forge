from contextlib import contextmanager
from dataclasses import dataclass
import torch
import torch._subclasses.functional_tensor
import torch.fx.traceback as fx_traceback
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._functorch.utils import exposed_in
from torch._higher_order_ops.utils import autograd_not_implemented
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import _get_current_dispatch_mode
@cond_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def cond_op_dense(pred, true_fn, false_fn, operands):
    mode = _get_current_dispatch_mode()
    assert mode is None, 'Mode should never be enabled for CPU/CUDA key'
    if pred:
        return true_fn(*operands)
    else:
        return false_fn(*operands)