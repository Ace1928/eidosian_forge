import builtins
import sys
def getreader(encoding):
    """ Lookup up the codec for the given encoding and return
        its StreamReader class or factory function.

        Raises a LookupError in case the encoding cannot be found.

    """
    return lookup(encoding).streamreader