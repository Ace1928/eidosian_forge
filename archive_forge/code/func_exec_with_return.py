from __future__ import annotations
import ast
import base64
import copy
import io
import pathlib
import pkgutil
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from html import escape
from textwrap import dedent
from typing import Any, Dict, List
import markdown
def exec_with_return(code: str, global_context: Dict[str, Any]=None, stdout: Any=None, stderr: Any=None) -> Any:
    """
    Executes a code snippet and returns the resulting output of the
    last line.

    Arguments
    ---------
    code: str
        The code to execute
    global_context: Dict[str, Any]
        The globals to inject into the execution context.
    stdout: io.StringIO
        The stream to redirect stdout to.
    stderr: io.StringIO
        The stream to redirect stderr to.

    Returns
    -------

    The return value of the executed code.
    """
    global_context = global_context if global_context else globals()
    global_context['display'] = _display
    code_ast = ast.parse(code)
    init_ast = copy.deepcopy(code_ast)
    init_ast.body = code_ast.body[:-1]
    last_ast = copy.deepcopy(code_ast)
    last_ast.body = code_ast.body[-1:]
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            exec(compile(init_ast, '<ast>', 'exec'), global_context)
            if not last_ast.body:
                out = None
            elif type(last_ast.body[0]) == ast.Expr:
                out = eval(compile(_convert_expr(last_ast.body[0]), '<ast>', 'eval'), global_context)
            else:
                exec(compile(last_ast, '<ast>', 'exec'), global_context)
                out = None
            if code.strip().endswith(';'):
                out = None
            if _OUT_BUFFER and out is None:
                out = _OUT_BUFFER[-1]
        except Exception:
            out = None
            traceback.print_exc(file=stderr)
        finally:
            _OUT_BUFFER.clear()
    return out