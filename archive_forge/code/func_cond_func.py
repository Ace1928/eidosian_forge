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
@cond_op.py_functionalize_impl
def cond_func(ctx, pred, true_fn, false_fn, inputs):
    unwrapped_inputs = ctx.unwrap_tensors(inputs)
    unwrapped_pred = ctx.unwrap_tensors(pred)
    with ctx.redispatch_to_next():
        functional_true = ctx.functionalize(true_fn)
        functional_false = ctx.functionalize(false_fn)
        for branch in [functional_true, functional_false]:
            if _has_potential_branch_input_mutation(branch, unwrapped_inputs):
                raise UnsupportedAliasMutationException('One of torch.cond branch might be modifying the input!')
        for branch in [true_fn, false_fn]:
            if _has_potential_branch_input_alias(branch, unwrapped_inputs):
                raise UnsupportedAliasMutationException('One of torch.cond branch might be aliasing the input!')
        cond_return = cond_op(unwrapped_pred, functional_true, functional_false, unwrapped_inputs)
        return ctx.wrap_tensors(cond_return)