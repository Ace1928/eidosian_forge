import re
import sys
import typing
from .util import (
from .unicode import pyparsing_unicode as ppu
@staticmethod
def explain_exception(exc, depth=16):
    """
        Method to take an exception and translate the Python internal traceback into a list
        of the pyparsing expressions that caused the exception to be raised.

        Parameters:

        - exc - exception raised during parsing (need not be a ParseException, in support
          of Python exceptions that might be raised in a parse action)
        - depth (default=16) - number of levels back in the stack trace to list expression
          and function names; if None, the full stack trace names will be listed; if 0, only
          the failing input line, marker, and exception string will be shown

        Returns a multi-line string listing the ParserElements and/or function names in the
        exception's stack trace.
        """
    import inspect
    from .core import ParserElement
    if depth is None:
        depth = sys.getrecursionlimit()
    ret = []
    if isinstance(exc, ParseBaseException):
        ret.append(exc.line)
        ret.append(' ' * (exc.column - 1) + '^')
    ret.append(f'{type(exc).__name__}: {exc}')
    if depth > 0:
        callers = inspect.getinnerframes(exc.__traceback__, context=depth)
        seen = set()
        for i, ff in enumerate(callers[-depth:]):
            frm = ff[0]
            f_self = frm.f_locals.get('self', None)
            if isinstance(f_self, ParserElement):
                if not frm.f_code.co_name.startswith(('parseImpl', '_parseNoCache')):
                    continue
                if id(f_self) in seen:
                    continue
                seen.add(id(f_self))
                self_type = type(f_self)
                ret.append(f'{self_type.__module__}.{self_type.__name__} - {f_self}')
            elif f_self is not None:
                self_type = type(f_self)
                ret.append(f'{self_type.__module__}.{self_type.__name__}')
            else:
                code = frm.f_code
                if code.co_name in ('wrapper', '<module>'):
                    continue
                ret.append(code.co_name)
            depth -= 1
            if not depth:
                break
    return '\n'.join(ret)