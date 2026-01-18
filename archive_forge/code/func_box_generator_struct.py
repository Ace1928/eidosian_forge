import llvmlite.ir
from llvmlite.ir import Constant, IRBuilder
from numba.core import types, config, cgutils
from numba.core.funcdesc import FunctionDescriptor
def box_generator_struct(self, lower, gen_struct):
    """
        Box the raw *gen_struct* as a Python object.
        """
    gen_ptr = cgutils.alloca_once_value(lower.builder, gen_struct)
    return lower.pyapi.from_native_generator(gen_ptr, self.gentype, lower.envarg)