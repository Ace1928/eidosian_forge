import sys
import typing as t
from types import CodeType
from types import TracebackType
from .exceptions import TemplateSyntaxError
from .utils import internal_code
from .utils import missing
def fake_traceback(exc_value: BaseException, tb: t.Optional[TracebackType], filename: str, lineno: int) -> TracebackType:
    """Produce a new traceback object that looks like it came from the
    template source instead of the compiled code. The filename, line
    number, and location name will point to the template, and the local
    variables will be the current template context.

    :param exc_value: The original exception to be re-raised to create
        the new traceback.
    :param tb: The original traceback to get the local variables and
        code info from.
    :param filename: The template filename.
    :param lineno: The line number in the template source.
    """
    if tb is not None:
        locals = get_template_locals(tb.tb_frame.f_locals)
        locals.pop('__jinja_exception__', None)
    else:
        locals = {}
    globals = {'__name__': filename, '__file__': filename, '__jinja_exception__': exc_value}
    code: CodeType = compile('\n' * (lineno - 1) + 'raise __jinja_exception__', filename, 'exec')
    location = 'template'
    if tb is not None:
        function = tb.tb_frame.f_code.co_name
        if function == 'root':
            location = 'top-level template code'
        elif function.startswith('block_'):
            location = f'block {function[6:]!r}'
    if sys.version_info >= (3, 8):
        code = code.replace(co_name=location)
    else:
        code = CodeType(code.co_argcount, code.co_kwonlyargcount, code.co_nlocals, code.co_stacksize, code.co_flags, code.co_code, code.co_consts, code.co_names, code.co_varnames, code.co_filename, location, code.co_firstlineno, code.co_lnotab, code.co_freevars, code.co_cellvars)
    try:
        exec(code, globals, locals)
    except BaseException:
        return sys.exc_info()[2].tb_next