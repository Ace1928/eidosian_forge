import fnmatch
import sys
import os
from inspect import CO_GENERATOR, CO_COROUTINE, CO_ASYNC_GENERATOR
def bpprint(self, out=None):
    """Print the output of bpformat().

        The optional out argument directs where the output is sent
        and defaults to standard output.
        """
    if out is None:
        out = sys.stdout
    print(self.bpformat(), file=out)