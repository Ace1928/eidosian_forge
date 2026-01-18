import sys
from typing import Dict, List, Optional, Type, overload
def is_simple_tuple(value: ast.AST) -> bool:
    return isinstance(value, ast.Tuple) and bool(value.elts) and (not any((isinstance(elt, ast.Starred) for elt in value.elts)))