from cupyx.jit import _internal_types
from cupyx.jit import _cuda_types
from cupyx.jit._internal_types import Data as _Data
@_wrap_thrust_func(['thrust/reverse.h'])
def reverse_copy(env, exec_policy, first, last, result):
    """Reverses a range.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_pointer_type(first)
    _assert_same_type(first, last)
    _assert_pointer_type(result)
    args = [exec_policy, first, last, result]
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::reverse_copy({params})', result.ctype)