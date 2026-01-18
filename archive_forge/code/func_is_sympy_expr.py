from __future__ import annotations
import re
import sys
from typing import (
import param  # type: ignore
from pyviz_comms import Comm, JupyterComm  # type: ignore
from ..io.resources import CDN_DIST
from ..util import lazy_load
from .base import ModelPane
def is_sympy_expr(obj: Any) -> bool:
    """Test for sympy.Expr types without usually needing to import sympy"""
    if 'sympy' in sys.modules and 'sympy' in str(type(obj).__class__):
        import sympy
        if isinstance(obj, sympy.Expr):
            return True
    return False