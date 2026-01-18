from cupyx.jit import _internal_types
from cupyx.jit import _cuda_types
from cupyx.jit._internal_types import Data as _Data
@_wrap_thrust_func(['thrust/sort.h'])
def sort_by_key(env, exec_policy, keys_first, keys_last, values_first, comp=None):
    """Performs key-value sort.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_pointer_type(keys_first)
    _assert_same_type(keys_first, keys_last)
    _assert_pointer_type(values_first)
    if comp is not None:
        raise NotImplementedError('comp option is not supported')
    args = [exec_policy, keys_first, keys_last, values_first]
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::sort_by_key({params})', _cuda_types.void)