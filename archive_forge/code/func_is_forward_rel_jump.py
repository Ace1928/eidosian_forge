import dis
import enum
import opcode as _opcode
import sys
from abc import abstractmethod
from dataclasses import dataclass
from marshal import dumps as _dumps
from typing import Any, Callable, Dict, Generic, Optional, Tuple, TypeVar, Union
import bytecode as _bytecode
def is_forward_rel_jump(self) -> bool:
    """Is a forward relative jump."""
    return self._opcode in _opcode.hasjrel and 'BACKWARD' not in self._name