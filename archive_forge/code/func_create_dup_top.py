import copy
import dataclasses
import dis
import itertools
import sys
import types
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Tuple
from .bytecode_analysis import (
def create_dup_top() -> Instruction:
    if sys.version_info >= (3, 11):
        return create_instruction('COPY', arg=1)
    return create_instruction('DUP_TOP')