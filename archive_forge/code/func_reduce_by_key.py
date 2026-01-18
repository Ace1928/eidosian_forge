from cupyx.jit import _internal_types
from cupyx.jit import _cuda_types
from cupyx.jit._internal_types import Data as _Data
@_wrap_thrust_func(['thrust/reduce.h'])
def reduce_by_key(env, exec_policy, keys_first, keys_last, values_first, keys_output, values_output, binary_pred=None, binary_op=None):
    """Generalization of reduce to key-value pairs.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_pointer_type(keys_first)
    _assert_same_type(keys_first, keys_last)
    _assert_pointer_type(values_first)
    _assert_pointer_type(keys_output)
    _assert_pointer_type(values_output)
    args = [exec_policy, keys_first, keys_last, values_first, keys_output, values_output]
    if binary_pred is not None:
        raise NotImplementedError('binary_pred option is not supported')
    if binary_op is not None:
        raise NotImplementedError('binary_op option is not supported')
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::reduce_by_key({params})', _cuda_types.Tuple([keys_output.ctype, values_output.ctype]))