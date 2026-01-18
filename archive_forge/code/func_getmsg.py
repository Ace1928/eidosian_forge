import py
import sys, inspect
from compiler import parse, ast, pycodegen
from py._code.assertion import BuiltinAssertionError, _format_explanation
import types
def getmsg(excinfo):
    if isinstance(excinfo, tuple):
        excinfo = py.code.ExceptionInfo(excinfo)
    tb = excinfo.traceback[-1]
    source = str(tb.statement).strip()
    x = interpret(source, tb.frame, should_fail=True)
    if not isinstance(x, str):
        raise TypeError('interpret returned non-string %r' % (x,))
    return x