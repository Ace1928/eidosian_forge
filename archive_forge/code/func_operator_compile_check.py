import torch
from .make_fx import make_fx_check
from .aot_autograd import aot_autograd_check
from .fake_tensor import fake_check
from torch._subclasses.schema_check_mode import SchemaCheckMode
def operator_compile_check(func, args, kwargs=None, *, dynamic_only=False, supports_autograd=True, fullgraph=True):
    """Check if torch.compile supports a custom operator.

    Args:
        func (function): an operator that takes at least one Tensor as input
            and returns a Tensor or a Tuple of Tensors.
        args (Tuple): args to the operator
        kwargs (dict, optional): kwargs to the operator
        dynamic_only (bool, optional): If the operator only works with dynamic
            shapes. This can happen if it returns Tensors whose shape are
            dependent on the data on the input Tensors. If True, we skip
            tests related to torch.compile with static shapes.
        supports_autograd (bool, optional): If the operator does not support
            autograd. If False, we will skip autograd-related tests.
        fullgraph (bool, optional): If we expect torch.compile to not graph
            break on seeing the operator. Graph breaks are bad for performance.

    """
    if kwargs is None:
        kwargs = {}
    func = add_clones(func)

    def run_static_or_dynamic_tests(dynamic):
        tracing_mode = 'symbolic' if dynamic else 'fake'
        make_fx_check(func, args, kwargs, tracing_mode=tracing_mode)
        if supports_autograd:
            aot_autograd_check(func, args, kwargs, dynamic=dynamic)
        check_compile(func, args, kwargs, fullgraph=fullgraph, backend='aot_eager', dynamic=dynamic)
        check_compile(func, args, kwargs, fullgraph=fullgraph, backend='inductor', dynamic=dynamic)
    schema_check(func, args, kwargs)
    fake_check(func, args, kwargs, dynamic_only)
    if not dynamic_only:
        run_static_or_dynamic_tests(dynamic=False)
    run_static_or_dynamic_tests(dynamic=True)