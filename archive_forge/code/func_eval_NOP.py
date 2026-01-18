import opcode
import operator
import sys
from _pydevd_frame_eval.vendored.bytecode import Instr, Bytecode, ControlFlowGraph, BasicBlock, Compare
def eval_NOP(self, instr):
    del self.block[self.index - 1]
    self.index -= 1