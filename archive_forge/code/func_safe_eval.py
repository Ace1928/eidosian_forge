import ast
import sys
import builtins
from typing import Dict, Any, Optional
from . import line as line_properties
from .inspection import getattr_safe
def safe_eval(expr: str, namespace: Dict[str, Any]) -> Any:
    """Not all that safe, just catches some errors"""
    try:
        return eval(expr, namespace)
    except (NameError, AttributeError, SyntaxError):
        raise EvaluationError