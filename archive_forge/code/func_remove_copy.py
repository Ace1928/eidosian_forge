from cupyx.jit import _internal_types
from cupyx.jit import _cuda_types
from cupyx.jit._internal_types import Data as _Data
@_wrap_thrust_func(['thrust/remove.h'])
def remove_copy(env, exec_policy, first, last, result, value):
    """Removes from the range all elements that are rqual to value.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_pointer_type(first)
    _assert_same_type(first, last)
    _assert_pointer_type(result)
    args = [exec_policy, first, last, result, value]
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::remove_copy({params})', result.ctype)