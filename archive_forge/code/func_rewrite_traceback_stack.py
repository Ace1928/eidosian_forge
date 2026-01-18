import sys
import typing as t
from types import CodeType
from types import TracebackType
from .exceptions import TemplateSyntaxError
from .utils import internal_code
from .utils import missing
def rewrite_traceback_stack(source: t.Optional[str]=None) -> BaseException:
    """Rewrite the current exception to replace any tracebacks from
    within compiled template code with tracebacks that look like they
    came from the template source.

    This must be called within an ``except`` block.

    :param source: For ``TemplateSyntaxError``, the original source if
        known.
    :return: The original exception with the rewritten traceback.
    """
    _, exc_value, tb = sys.exc_info()
    exc_value = t.cast(BaseException, exc_value)
    tb = t.cast(TracebackType, tb)
    if isinstance(exc_value, TemplateSyntaxError) and (not exc_value.translated):
        exc_value.translated = True
        exc_value.source = source
        exc_value.with_traceback(None)
        tb = fake_traceback(exc_value, None, exc_value.filename or '<unknown>', exc_value.lineno)
    else:
        tb = tb.tb_next
    stack = []
    while tb is not None:
        if tb.tb_frame.f_code in internal_code:
            tb = tb.tb_next
            continue
        template = tb.tb_frame.f_globals.get('__jinja_template__')
        if template is not None:
            lineno = template.get_corresponding_lineno(tb.tb_lineno)
            fake_tb = fake_traceback(exc_value, tb, template.filename, lineno)
            stack.append(fake_tb)
        else:
            stack.append(tb)
        tb = tb.tb_next
    tb_next = None
    for tb in reversed(stack):
        tb.tb_next = tb_next
        tb_next = tb
    return exc_value.with_traceback(tb_next)