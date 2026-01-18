import llvmlite.ir
from llvmlite.ir import Constant, IRBuilder
from numba.core import types, config, cgutils
from numba.core.funcdesc import FunctionDescriptor
def return_from_generator(self, lower):
    """
        Emit a StopIteration at generator end and mark the generator exhausted.
        """
    indexval = Constant(self.resume_index_ptr.type.pointee, -1)
    lower.builder.store(indexval, self.resume_index_ptr)
    self.call_conv.return_stop_iteration(lower.builder)