import sys
from _pydevd_frame_eval.vendored import bytecode as _bytecode
from _pydevd_frame_eval.vendored.bytecode.concrete import ConcreteInstr
from _pydevd_frame_eval.vendored.bytecode.flags import CompilerFlags
from _pydevd_frame_eval.vendored.bytecode.instr import Label, SetLineno, Instr
def get_jump(self):
    if not self:
        return None
    last_instr = self[-1]
    if not (isinstance(last_instr, Instr) and last_instr.has_jump()):
        return None
    target_block = last_instr.arg
    assert isinstance(target_block, BasicBlock)
    return target_block