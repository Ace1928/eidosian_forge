import llvmlite.ir
from llvmlite.ir import Constant, IRBuilder
from numba.core import types, config, cgutils
from numba.core.funcdesc import FunctionDescriptor
def get_resume_index_ptr(self, builder, genptr):
    return cgutils.gep_inbounds(builder, genptr, 0, 0, name='gen.resume_index')