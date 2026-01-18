import copy
import dataclasses
import dis
import itertools
import sys
import types
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Tuple
from .bytecode_analysis import (
def instruction_size(inst) -> int:
    if sys.version_info >= (3, 11):
        return 2 * (_PYOPCODE_CACHES.get(dis.opname[inst.opcode], 0) + 1)
    return 2