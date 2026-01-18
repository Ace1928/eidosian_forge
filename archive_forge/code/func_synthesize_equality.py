import ast
import dataclasses
import inspect
import os
from functools import partial
from typing import Callable, Dict, List
from torch._jit_internal import FAKE_FILENAME_PREFIX, is_optional
from torch._sources import ParsedDef, SourceContext
def synthesize_equality(cls, name: str, converse: str) -> ParsedDef:
    return synthesize_comparison(cls, name, allow_eq=True, raise_on_none=False, inner=[f'if val1 {converse} val2: return False'])