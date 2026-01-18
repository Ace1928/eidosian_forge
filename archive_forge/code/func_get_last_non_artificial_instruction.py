import sys
import types
from collections import defaultdict
from dataclasses import dataclass
from typing import (
import bytecode as _bytecode
from bytecode.concrete import ConcreteInstr
from bytecode.flags import CompilerFlags
from bytecode.instr import UNSET, Instr, Label, SetLineno, TryBegin, TryEnd
def get_last_non_artificial_instruction(self) -> Optional[Instr]:
    for instr in reversed(self):
        if isinstance(instr, Instr):
            return instr
    return None