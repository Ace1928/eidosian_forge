from cupyx.jit import _internal_types
from cupyx.jit import _cuda_types
from cupyx.jit._internal_types import Data as _Data
@_wrap_thrust_func(['thrust/adjacent_difference.h'])
def adjacent_difference(env, exec_policy, first, last, result, binary_op=None):
    """Computes the differences of adjacent elements.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_same_type(first, last)
    _assert_same_pointer_type(first, result)
    if binary_op is not None:
        raise NotImplementedError('binary_op option is not supported')
    args = [exec_policy, first, last, result]
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::adjacent_difference({params})', result.ctype)