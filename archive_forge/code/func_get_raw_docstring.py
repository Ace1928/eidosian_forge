import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
def get_raw_docstring(self, node):
    """If a docstring node is found in the body of the *node* parameter,
        return that docstring node, None otherwise.

        Logic mirrored from ``_PyAST_GetDocString``."""
    if not isinstance(node, (AsyncFunctionDef, FunctionDef, ClassDef, Module)) or len(node.body) < 1:
        return None
    node = node.body[0]
    if not isinstance(node, Expr):
        return None
    node = node.value
    if isinstance(node, Constant) and isinstance(node.value, str):
        return node