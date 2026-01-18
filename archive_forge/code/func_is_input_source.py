import collections
import dataclasses
import enum
from typing import Any, Optional, Union
from torch._guards import ChainedSource, GuardSource, Source
from . import utils
from .bytecode_transformation import create_call_function, create_instruction
from .utils import enum_repr
def is_input_source(source):
    return source.guard_source() in [GuardSource.LOCAL, GuardSource.GLOBAL, GuardSource.LOCAL_NN_MODULE, GuardSource.GLOBAL_NN_MODULE, GuardSource.LOCAL_FSDP_MODULE, GuardSource.GLOBAL_FSDP_MODULE]