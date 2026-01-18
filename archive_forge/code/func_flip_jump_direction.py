import copy
import dataclasses
import dis
import itertools
import sys
import types
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Tuple
from .bytecode_analysis import (
def flip_jump_direction(instruction: Instruction) -> None:
    if sys.version_info < (3, 11):
        raise RuntimeError('Cannot flip jump direction in Python < 3.11')
    if 'FORWARD' in instruction.opname:
        instruction.opname = instruction.opname.replace('FORWARD', 'BACKWARD')
    elif 'BACKWARD' in instruction.opname:
        instruction.opname = instruction.opname.replace('BACKWARD', 'FORWARD')
    else:
        raise AttributeError('Instruction is not a forward or backward jump')
    instruction.opcode = dis.opmap[instruction.opname]
    assert instruction.opcode in _REL_JUMPS