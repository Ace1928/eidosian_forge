from cupyx.jit import _internal_types
from cupyx.jit import _cuda_types
from cupyx.jit._internal_types import Data as _Data
@_wrap_thrust_func(['thrust/sort.h'])
def stable_sort(env, exec_policy, first, last, comp=None):
    """Sorts the elements in [first, last) into ascending order.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_pointer_type(first)
    _assert_same_type(first, last)
    if comp is not None:
        raise NotImplementedError('comp option is not supported')
    args = [exec_policy, first, last]
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::stable_sort({params})', _cuda_types.void)