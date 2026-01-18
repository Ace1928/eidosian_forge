from cupyx.jit import _internal_types
from cupyx.jit import _cuda_types
from cupyx.jit._internal_types import Data as _Data
@_wrap_thrust_func(['thrust/set_operations.h'])
def set_symmetric_difference(env, exec_policy, first1, last1, first2, last2, result, comp=None):
    """Constructs a sorted range that is the symmetric difference.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_same_type(first1, last1)
    _assert_same_type(first2, last2)
    _assert_pointer_type(result)
    args = [exec_policy, first1, last1, first2, last2, result]
    if comp is not None:
        raise NotImplementedError('comp option is not supported')
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::set_symmetric_difference({params})', result.ctype)