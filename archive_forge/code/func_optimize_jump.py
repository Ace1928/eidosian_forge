import opcode
import operator
import sys
from _pydevd_frame_eval.vendored.bytecode import Instr, Bytecode, ControlFlowGraph, BasicBlock, Compare
def optimize_jump(self, instr):
    if instr.is_uncond_jump() and self.index == len(self.block):
        block_index = self.block_index
        target_block = instr.arg
        target_block_index = self.code.get_block_index(target_block)
        if target_block_index == block_index:
            del self.block[self.index - 1]
            self.block.next_block = target_block
            return
    self.optimize_jump_to_cond_jump(instr)