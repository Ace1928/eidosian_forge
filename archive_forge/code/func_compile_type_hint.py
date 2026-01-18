from __future__ import annotations
from ast import (
from ast import Tuple as ASTTuple
from types import CodeType
from typing import Any, Dict, FrozenSet, List, Set, Tuple, Union
def compile_type_hint(hint: str) -> CodeType:
    parsed = parse(hint, '<string>', 'eval')
    UnionTransformer().visit(parsed)
    fix_missing_locations(parsed)
    return compile(parsed, '<string>', 'eval', flags=0)