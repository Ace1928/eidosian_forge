import collections
import dataclasses
import enum
from typing import Any, Optional, Union
from torch._guards import ChainedSource, GuardSource, Source
from . import utils
from .bytecode_transformation import create_call_function, create_instruction
from .utils import enum_repr
def is_constant_source(source):
    if isinstance(source, ConstantSource):
        return True
    try:
        if source.guard_source() == GuardSource.CONSTANT:
            return True
    except NotImplementedError:
        pass
    return False