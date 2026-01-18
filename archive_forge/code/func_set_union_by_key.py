from cupyx.jit import _internal_types
from cupyx.jit import _cuda_types
from cupyx.jit._internal_types import Data as _Data
@_wrap_thrust_func(['thrust/set_operations.h'])
def set_union_by_key(env, exec_policy, keys_first1, keys_last1, keys_first2, keys_last2, values_first1, values_first2, keys_result, values_result, comp=None):
    """Constructs the key-value union of sorted inputs.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_same_type(keys_first1, keys_last1)
    _assert_same_type(keys_first2, keys_last2)
    _assert_pointer_type(values_first1)
    _assert_pointer_type(values_first2)
    _assert_pointer_type(keys_result)
    _assert_pointer_type(values_result)
    args = [exec_policy, keys_first1, keys_last1, keys_first2, keys_last2, values_first1, values_first2, keys_result, values_result]
    if comp is not None:
        raise NotImplementedError('comp option is not supported')
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::set_union_by_key({params})', _cuda_types.Tuple([keys_result.ctype, values_result.ctype]))