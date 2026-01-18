from __future__ import annotations
import ast
import builtins
import sys
import typing
from ast import (
from collections import defaultdict
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, ClassVar, cast, overload
def name_matches(self, expression: expr | Expr | None, *names: str) -> bool:
    if expression is None:
        return False
    path: list[str] = []
    top_expression = expression.value if isinstance(expression, Expr) else expression
    if isinstance(top_expression, Subscript):
        top_expression = top_expression.value
    elif isinstance(top_expression, Call):
        top_expression = top_expression.func
    while isinstance(top_expression, Attribute):
        path.insert(0, top_expression.attr)
        top_expression = top_expression.value
    if not isinstance(top_expression, Name):
        return False
    if top_expression.id in self.imported_names:
        translated = self.imported_names[top_expression.id]
    elif hasattr(builtins, top_expression.id):
        translated = 'builtins.' + top_expression.id
    else:
        translated = top_expression.id
    path.insert(0, translated)
    joined_path = '.'.join(path)
    if joined_path in names:
        return True
    elif self.parent:
        return self.parent.name_matches(expression, *names)
    else:
        return False