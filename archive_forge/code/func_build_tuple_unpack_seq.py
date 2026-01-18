import opcode
import operator
import sys
from _pydevd_frame_eval.vendored.bytecode import Instr, Bytecode, ControlFlowGraph, BasicBlock, Compare
def build_tuple_unpack_seq(self, instr):
    next_instr = self.get_next_instr('UNPACK_SEQUENCE')
    if next_instr is None or next_instr.arg != instr.arg:
        return
    if instr.arg < 1:
        return
    if self.const_stack and instr.arg <= len(self.const_stack):
        nconst = instr.arg
        start = self.index - 1
        load_consts = self.block[start - nconst:start]
        self.block[start - nconst:start] = reversed(load_consts)
        self.block[start:start + 2] = ()
        self.index -= 2
        self.const_stack.clear()
        return
    if instr.arg == 1:
        del self.block[self.index - 1:self.index + 1]
    elif instr.arg == 2:
        rot2 = Instr('ROT_TWO', lineno=instr.lineno)
        self.block[self.index - 1:self.index + 1] = (rot2,)
        self.index -= 1
        self.const_stack.clear()
    elif instr.arg == 3:
        rot3 = Instr('ROT_THREE', lineno=instr.lineno)
        rot2 = Instr('ROT_TWO', lineno=instr.lineno)
        self.block[self.index - 1:self.index + 1] = (rot3, rot2)
        self.index -= 1
        self.const_stack.clear()