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
def macro_def(self, macro_ref: MacroRef, frame: Frame) -> None:
    """Dump the macro definition for the def created by macro_body."""
    arg_tuple = ', '.join((repr(x.name) for x in macro_ref.node.args))
    name = getattr(macro_ref.node, 'name', None)
    if len(macro_ref.node.args) == 1:
        arg_tuple += ','
    self.write(f'Macro(environment, macro, {name!r}, ({arg_tuple}), {macro_ref.accesses_kwargs!r}, {macro_ref.accesses_varargs!r}, {macro_ref.accesses_caller!r}, context.eval_ctx.autoescape)')