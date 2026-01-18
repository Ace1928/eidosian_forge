import ast
import inspect
import re
import sys
import warnings
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union
from .. import language
from .._C.libtriton.triton import ir
from ..language import constexpr, tensor
from ..runtime import JITFunction
from .errors import (CompilationError, CompileTimeAssertionFailure, UnsupportedLanguageConstruct)
def execute_static_print(self, node: ast.Call) -> None:
    kws = {name: _unwrap_if_constexpr(value) for name, value in (self.visit(keyword) for keyword in node.keywords)}
    args = [_unwrap_if_constexpr(self.visit(arg)) for arg in node.args]
    print(*args, **kws)