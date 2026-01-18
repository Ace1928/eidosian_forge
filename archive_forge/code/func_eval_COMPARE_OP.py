import opcode
import operator
import sys
from _pydevd_frame_eval.vendored.bytecode import Instr, Bytecode, ControlFlowGraph, BasicBlock, Compare
def eval_COMPARE_OP(self, instr):
    try:
        new_arg = NOT_COMPARE[instr.arg]
    except KeyError:
        return
    if self.get_next_instr('UNARY_NOT') is None:
        return
    instr.arg = new_arg
    self.block[self.index - 1:self.index + 1] = (instr,)