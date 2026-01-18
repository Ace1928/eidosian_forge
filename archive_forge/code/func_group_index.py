from cupy.cuda import runtime as _runtime
from cupyx.jit import _compile
from cupyx.jit import _cuda_types
from cupyx.jit._internal_types import BuiltinFunc as _BuiltinFunc
from cupyx.jit._internal_types import Constant as _Constant
from cupyx.jit._internal_types import Data as _Data
from cupyx.jit._internal_types import wraps_class_method as _wraps_class_method
@_wraps_class_method
def group_index(self, env, instance):
    """
        group_index()

        3-Dimensional index of the block within the launched grid.
        """
    _check_include(env, 'cg')
    return _Data(f'{instance.code}.group_index()', _cuda_types.dim3)