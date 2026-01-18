from cupyx.jit import _internal_types
from cupyx.jit import _cuda_types
from cupyx.jit._internal_types import Data as _Data
@_wrap_thrust_func(['thrust/scan.h'])
def exclusive_scan_by_key(env, exec_policy, first1, last1, first2, result, init=None, binary_pred=None, binary_op=None):
    """Computes an exclusive prefix sum operation by key.
    _assert_exec_policy_type(exec_policy)
    """
    _assert_pointer_type(first1)
    _assert_same_type(first1, last1)
    _assert_same_pointer_type(first2, result)
    if binary_pred is not None:
        raise NotImplementedError('binary_pred option is not supported')
    if binary_op is not None:
        raise NotImplementedError('binary_op option is not supported')
    args = [exec_policy, first1, last1, first2, result]
    if init is not None:
        args.append(init)
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::exclusive_scan_by_key({params})', result.ctype)