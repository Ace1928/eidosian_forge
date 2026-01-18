import llvmlite.ir
from llvmlite.ir import Constant, IRBuilder
from numba.core import types, config, cgutils
from numba.core.funcdesc import FunctionDescriptor
def lower_yield_resume(self):
    self.genlower.create_resumption_block(self.lower, self.inst.index)
    self.lower.debug_print('# generator resume')
    for state_index, name in zip(self.live_var_indices, self.live_vars):
        state_slot = cgutils.gep_inbounds(self.builder, self.gen_state_ptr, 0, state_index)
        ty = self.gentype.state_types[state_index]
        val = self.context.unpack_value(self.builder, ty, state_slot)
        self.lower.storevar(val, name)
        if self.context.enable_nrt:
            self.context.nrt.decref(self.builder, ty, val)
    self.lower.debug_print('# generator resume end')