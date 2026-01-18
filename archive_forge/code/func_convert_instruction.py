import copy
import dataclasses
import dis
import itertools
import sys
import types
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Tuple
from .bytecode_analysis import (
def convert_instruction(i: dis.Instruction) -> Instruction:
    return Instruction(i.opcode, i.opname, i.arg, i.argval, i.offset, i.starts_line, i.is_jump_target, getattr(i, 'positions', None))