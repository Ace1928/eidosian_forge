import llvmlite.ir
from llvmlite.ir import Constant, IRBuilder
from numba.core import types, config, cgutils
from numba.core.funcdesc import FunctionDescriptor
def lower_finalize_func_body(self, builder, genptr):
    """
        Lower the body of the generator's finalizer: decref all live
        state variables.
        """
    pyapi = self.context.get_python_api(builder)
    resume_index_ptr = self.get_resume_index_ptr(builder, genptr)
    resume_index = builder.load(resume_index_ptr)
    need_cleanup = builder.icmp_signed('>', resume_index, Constant(resume_index.type, 0))
    with cgutils.if_unlikely(builder, need_cleanup):
        gen_state_ptr = self.get_state_ptr(builder, genptr)
        for state_index in range(len(self.gentype.state_types)):
            state_slot = cgutils.gep_inbounds(builder, gen_state_ptr, 0, state_index)
            ty = self.gentype.state_types[state_index]
            val = self.context.unpack_value(builder, ty, state_slot)
            pyapi.decref(val)
    builder.ret_void()