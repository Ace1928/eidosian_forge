import llvmlite.ir
from llvmlite.ir import Constant, IRBuilder
from numba.core import types, config, cgutils
from numba.core.funcdesc import FunctionDescriptor
def init_generator_state(self, lower):
    """
        NULL-initialize all generator state variables, to avoid spurious
        decref's on cleanup.
        """
    lower.builder.store(Constant(self.gen_state_ptr.type.pointee, None), self.gen_state_ptr)