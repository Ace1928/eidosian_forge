import typing as t
from contextlib import contextmanager
from functools import update_wrapper
from io import StringIO
from itertools import chain
from keyword import iskeyword as is_python_keyword
from markupsafe import escape
from markupsafe import Markup
from . import nodes
from .exceptions import TemplateAssertionError
from .idtracking import Symbols
from .idtracking import VAR_LOAD_ALIAS
from .idtracking import VAR_LOAD_PARAMETER
from .idtracking import VAR_LOAD_RESOLVE
from .idtracking import VAR_LOAD_UNDEFINED
from .nodes import EvalContext
from .optimizer import Optimizer
from .utils import _PassArg
from .utils import concat
from .visitor import NodeVisitor
def enter_frame(self, frame: Frame) -> None:
    undefs = []
    for target, (action, param) in frame.symbols.loads.items():
        if action == VAR_LOAD_PARAMETER:
            pass
        elif action == VAR_LOAD_RESOLVE:
            self.writeline(f'{target} = {self.get_resolve_func()}({param!r})')
        elif action == VAR_LOAD_ALIAS:
            self.writeline(f'{target} = {param}')
        elif action == VAR_LOAD_UNDEFINED:
            undefs.append(target)
        else:
            raise NotImplementedError('unknown load instruction')
    if undefs:
        self.writeline(f'{' = '.join(undefs)} = missing')