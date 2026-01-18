import os
import re
from incremental import Version
from twisted.python.deprecate import deprecatedModuleAttribute
def cmdLineQuote(s):
    """
    Internal method for quoting a single command-line argument.

    @param s: an unquoted string that you want to quote so that something that
        does cmd.exe-style unquoting will interpret it as a single argument,
        even if it contains spaces.
    @type s: C{str}

    @return: a quoted string.
    @rtype: C{str}
    """
    quote = (' ' in s or '\t' in s or '"' in s or (s == '')) and '"' or ''
    return quote + _cmdLineQuoteRe2.sub('\\1\\1', _cmdLineQuoteRe.sub('\\1\\1\\\\"', s)) + quote