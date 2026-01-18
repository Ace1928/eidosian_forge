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
def return_buffer_contents(self, frame: Frame, force_unescaped: bool=False) -> None:
    """Return the buffer contents of the frame."""
    if not force_unescaped:
        if frame.eval_ctx.volatile:
            self.writeline('if context.eval_ctx.autoescape:')
            self.indent()
            self.writeline(f'return Markup(concat({frame.buffer}))')
            self.outdent()
            self.writeline('else:')
            self.indent()
            self.writeline(f'return concat({frame.buffer})')
            self.outdent()
            return
        elif frame.eval_ctx.autoescape:
            self.writeline(f'return Markup(concat({frame.buffer}))')
            return
    self.writeline(f'return concat({frame.buffer})')