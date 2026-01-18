import copy
import dataclasses
import dis
import itertools
import sys
import types
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Tuple
from .bytecode_analysis import (
def devirtualize_jumps(instructions):
    """Fill in args for virtualized jump target after instructions may have moved"""
    indexof = get_indexof(instructions)
    jumps = set(dis.hasjabs).union(set(dis.hasjrel))
    for inst in instructions:
        if inst.opcode in jumps:
            target = _get_instruction_front(instructions, indexof[inst.target])
            if inst.opcode in dis.hasjabs:
                if sys.version_info < (3, 10):
                    inst.arg = target.offset
                elif sys.version_info < (3, 11):
                    inst.arg = int(target.offset / 2)
                else:
                    raise RuntimeError('Python 3.11+ should not have absolute jumps')
            else:
                inst.arg = int(target.offset - inst.offset - instruction_size(inst))
                if inst.arg < 0:
                    if sys.version_info < (3, 11):
                        raise RuntimeError('Got negative jump offset for Python < 3.11')
                    inst.arg = -inst.arg
                    if 'FORWARD' in inst.opname:
                        flip_jump_direction(inst)
                elif inst.arg > 0:
                    if sys.version_info >= (3, 11) and 'BACKWARD' in inst.opname:
                        flip_jump_direction(inst)
                if sys.version_info >= (3, 10):
                    inst.arg //= 2
            inst.argval = target.offset
            inst.argrepr = f'to {target.offset}'