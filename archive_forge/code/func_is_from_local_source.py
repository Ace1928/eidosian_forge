import collections
import dataclasses
import enum
from typing import Any, Optional, Union
from torch._guards import ChainedSource, GuardSource, Source
from . import utils
from .bytecode_transformation import create_call_function, create_instruction
from .utils import enum_repr
def is_from_local_source(source: Source, *, allow_cell_or_freevar=True):
    if isinstance(source, ChainedSource):
        return is_from_local_source(source.base, allow_cell_or_freevar=allow_cell_or_freevar)
    if not isinstance(source, LocalSource):
        return False
    if not allow_cell_or_freevar and source.cell_or_freevar:
        return False
    return True