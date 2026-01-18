import copy
import dataclasses
import dis
import itertools
import sys
import types
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Tuple
from .bytecode_analysis import (
def remove_load_call_method(instructions: List[Instruction]) -> List[Instruction]:
    """LOAD_METHOD puts a NULL on the stack which causes issues, so remove it"""
    rewrites = {'LOAD_METHOD': 'LOAD_ATTR', 'CALL_METHOD': 'CALL_FUNCTION'}
    for inst in instructions:
        if inst.opname in rewrites:
            inst.opname = rewrites[inst.opname]
            inst.opcode = dis.opmap[inst.opname]
    return instructions