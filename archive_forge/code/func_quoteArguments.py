import os
import re
from incremental import Version
from twisted.python.deprecate import deprecatedModuleAttribute
def quoteArguments(arguments):
    """
    Quote an iterable of command-line arguments for passing to CreateProcess or
    a similar API.  This allows the list passed to C{reactor.spawnProcess} to
    match the child process's C{sys.argv} properly.

    @param arguments: an iterable of C{str}, each unquoted.

    @return: a single string, with the given sequence quoted as necessary.
    """
    return ' '.join([cmdLineQuote(a) for a in arguments])