from cupyx.jit import _internal_types
from cupyx.jit import _cuda_types
from cupyx.jit._internal_types import Data as _Data
@_wrap_thrust_func(['thrust/scan.h'])
def exclusive_scan(env, exec_policy, first, last, result, init=None, binary_op=None):
    """Computes an exclusive prefix sum operation.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_same_type(first, last)
    _assert_same_pointer_type(first, result)
    if binary_op is not None:
        raise NotImplementedError('binary_op option is not supported')
    args = [exec_policy, first, last, result]
    if init is not None:
        args.append(init)
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::exclusive_scan({params})', result.ctype)