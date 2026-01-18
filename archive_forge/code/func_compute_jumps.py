import dis
import inspect
import opcode as _opcode
import struct
import sys
import types
from _pydevd_frame_eval.vendored import bytecode as _bytecode
from _pydevd_frame_eval.vendored.bytecode.instr import (
def compute_jumps(self):
    offsets = []
    offset = 0
    for index, instr in enumerate(self.instructions):
        offsets.append(offset)
        offset += instr.size // 2 if OFFSET_AS_INSTRUCTION else instr.size
    offsets.append(offset)
    modified = False
    for index, label, instr in self.jumps:
        target_index = self.labels[label]
        target_offset = offsets[target_index]
        if instr.opcode in _opcode.hasjrel:
            instr_offset = offsets[index]
            target_offset -= instr_offset + (instr.size // 2 if OFFSET_AS_INSTRUCTION else instr.size)
        old_size = instr.size
        instr.arg = target_offset
        if instr.size != old_size:
            modified = True
    return modified