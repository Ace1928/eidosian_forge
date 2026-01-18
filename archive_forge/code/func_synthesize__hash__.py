import ast
import dataclasses
import inspect
import os
from functools import partial
from typing import Callable, Dict, List
from torch._jit_internal import FAKE_FILENAME_PREFIX, is_optional
from torch._sources import ParsedDef, SourceContext
def synthesize__hash__(cls) -> ParsedDef:
    return compose_fn(cls, '__hash__', ["raise NotImplementedError('__hash__ is not supported for dataclasses in TorchScript')"], signature='(self) -> int')