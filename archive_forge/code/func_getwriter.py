import builtins
import sys
def getwriter(encoding):
    """ Lookup up the codec for the given encoding and return
        its StreamWriter class or factory function.

        Raises a LookupError in case the encoding cannot be found.

    """
    return lookup(encoding).streamwriter