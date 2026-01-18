import code
import sys
import tokenize
from io import BytesIO
from traceback import format_exception
from types import TracebackType
from typing import Type
from twisted.conch import recvline
from twisted.internet import defer
from twisted.python.compat import _get_async_param
from twisted.python.htmlizer import TokenPrinter
from twisted.python.monkey import MonkeyPatcher
def lastColorizedLine(source):
    """
    Tokenize and colorize the given Python source.

    Returns a VT102-format colorized version of the last line of C{source}.

    @param source: Python source code
    @type source: L{str} or L{bytes}
    @return: L{bytes} of colorized source
    """
    if not isinstance(source, bytes):
        source = source.encode('utf-8')
    w = VT102Writer()
    p = TokenPrinter(w.write).printtoken
    s = BytesIO(source)
    for token in tokenize.tokenize(s.readline):
        tokenType, string, start, end, line = token
        p(tokenType, string, start, end, line)
    return bytes(w)