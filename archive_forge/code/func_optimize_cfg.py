import opcode
import operator
import sys
from _pydevd_frame_eval.vendored.bytecode import Instr, Bytecode, ControlFlowGraph, BasicBlock, Compare
def optimize_cfg(self, cfg):
    self.code = cfg
    self.const_stack = []
    self.remove_dead_blocks()
    self.block_index = 0
    while self.block_index < len(self.code):
        block = self.code[self.block_index]
        self.block_index += 1
        self.optimize_block(block)