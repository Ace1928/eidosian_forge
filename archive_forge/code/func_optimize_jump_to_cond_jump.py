import opcode
import operator
import sys
from _pydevd_frame_eval.vendored.bytecode import Instr, Bytecode, ControlFlowGraph, BasicBlock, Compare
def optimize_jump_to_cond_jump(self, instr):
    jump_label = instr.arg
    assert isinstance(jump_label, BasicBlock), jump_label
    try:
        target_instr = jump_label[0]
    except IndexError:
        return
    if instr.is_uncond_jump() and target_instr.name == 'RETURN_VALUE':
        self.block[self.index - 1] = target_instr
    elif target_instr.is_uncond_jump():
        jump_target2 = target_instr.arg
        name = instr.name
        if instr.name == 'JUMP_FORWARD':
            name = 'JUMP_ABSOLUTE'
        elif instr.opcode in opcode.hasjrel:
            return
        instr.name = name
        instr.arg = jump_target2
        self.block[self.index - 1] = instr