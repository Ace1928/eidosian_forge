from cupyx.jit import _internal_types
from cupyx.jit import _cuda_types
from cupyx.jit._internal_types import Data as _Data
@_wrap_thrust_func(['thrust/iterator/constant_iterator.h'])
def make_constant_iterator(env, x, i=None):
    """Finds the first positions whose values differ.
    """
    if i is not None:
        raise NotImplementedError('index_type is not supported')
    args = [x]
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::make_constant_iterator({params})', _ConstantIterator(x.ctype))