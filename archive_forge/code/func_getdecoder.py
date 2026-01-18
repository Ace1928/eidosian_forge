import builtins
import sys
def getdecoder(encoding):
    """ Lookup up the codec for the given encoding and return
        its decoder function.

        Raises a LookupError in case the encoding cannot be found.

    """
    return lookup(encoding).decode