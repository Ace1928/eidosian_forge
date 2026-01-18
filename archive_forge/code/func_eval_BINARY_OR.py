import opcode
import operator
import sys
from _pydevd_frame_eval.vendored.bytecode import Instr, Bytecode, ControlFlowGraph, BasicBlock, Compare
def eval_BINARY_OR(self, instr):
    return self.binop(operator.or_, instr)