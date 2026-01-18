import opcode
import operator
import sys
from _pydevd_frame_eval.vendored.bytecode import Instr, Bytecode, ControlFlowGraph, BasicBlock, Compare
def replace_container_of_consts(self, instr, container_type):
    items = self.const_stack[-instr.arg:]
    value = container_type(items)
    self.replace_load_const(instr.arg, instr, value)