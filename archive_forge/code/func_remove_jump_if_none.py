import copy
import dataclasses
import dis
import itertools
import sys
import types
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Tuple
from .bytecode_analysis import (
def remove_jump_if_none(instructions: List[Instruction]) -> None:
    new_insts = []
    for inst in instructions:
        new_insts.append(inst)
        if '_NONE' in inst.opname:
            is_op = create_instruction('IS_OP', arg=int('NOT' in inst.opname))
            is_op.argval = is_op.arg
            jump_op = create_instruction('POP_JUMP_FORWARD_IF_TRUE' if 'FORWARD' in inst.opname else 'POP_JUMP_BACKWARD_IF_TRUE', target=inst.target)
            inst.opcode = dis.opmap['LOAD_CONST']
            inst.opname = 'LOAD_CONST'
            inst.arg = None
            inst.argval = None
            new_insts.extend([is_op, jump_op])
    instructions[:] = new_insts