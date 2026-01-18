import ast
import dataclasses
import inspect
import os
from functools import partial
from typing import Callable, Dict, List
from torch._jit_internal import FAKE_FILENAME_PREFIX, is_optional
from torch._sources import ParsedDef, SourceContext
def synthesize_inequality(cls, name: str, op: str, allow_eq: bool) -> ParsedDef:
    return synthesize_comparison(cls, name, allow_eq, raise_on_none=True, inner=[f'if val1 {op} val2: return True', f'elif val2 {op} val1: return False'])