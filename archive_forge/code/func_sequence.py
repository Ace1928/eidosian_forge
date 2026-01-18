from cupyx.jit import _internal_types
from cupyx.jit import _cuda_types
from cupyx.jit._internal_types import Data as _Data
@_wrap_thrust_func(['thrust/sequence.h'])
def sequence(env, exec_policy, first, last, init=None, step=None):
    """Fills the range with a sequence of numbers.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_pointer_type(first)
    _assert_same_type(first, last)
    args = [exec_policy, first, last]
    if init is not None:
        args.append(init)
    if step is not None:
        _assert_same_type(init, step)
        args.append(step)
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::sequence({params})', _cuda_types.void)