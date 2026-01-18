from __future__ import annotations
import os
from typing import ClassVar
from ._typing import Literal
from ._utils import Parameters, _check_types, extract_parameters
from .exceptions import InvalidHashError
from .low_level import Type, hash_secret, verify_secret
from .profiles import RFC_9106_LOW_MEMORY
@property
def memory_cost(self) -> int:
    return self._parameters.memory_cost