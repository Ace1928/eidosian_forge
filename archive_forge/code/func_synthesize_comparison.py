import ast
import dataclasses
import inspect
import os
from functools import partial
from typing import Callable, Dict, List
from torch._jit_internal import FAKE_FILENAME_PREFIX, is_optional
from torch._sources import ParsedDef, SourceContext
def synthesize_comparison(cls, name: str, allow_eq: bool, raise_on_none: bool, inner: List[str]) -> ParsedDef:
    body = []
    for field in dataclasses.fields(cls):
        if not field.compare:
            continue
        body.extend([f'val1 = self.{field.name}', f'val2 = other.{field.name}'])
        body.extend(inner if not is_optional(field.type) else ['if val1 is not None and val2 is not None:', *['  ' + line for line in inner], 'elif (val1 is None) != (val2 is None):', f"  raise TypeError('Cannot compare {cls.__name__} with None')" if raise_on_none else '  return False'])
    body.append(f'return {allow_eq}')
    return compose_fn(cls, name, body, signature=f'(self, other: {cls.__name__}) -> bool')