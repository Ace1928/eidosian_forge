import enum
import dis
import opcode as _opcode
import sys
from marshal import dumps as _dumps
from _pydevd_frame_eval.vendored import bytecode as _bytecode
def is_uncond_jump(self):
    """Is an unconditional jump?"""
    return self.name in {'JUMP_FORWARD', 'JUMP_ABSOLUTE'}