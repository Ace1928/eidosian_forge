import fnmatch
import sys
import os
from inspect import CO_GENERATOR, CO_COROUTINE, CO_ASYNC_GENERATOR
def get_bpbynumber(self, arg):
    """Return a breakpoint by its index in Breakpoint.bybpnumber.

        For invalid arg values or if the breakpoint doesn't exist,
        raise a ValueError.
        """
    if not arg:
        raise ValueError('Breakpoint number expected')
    try:
        number = int(arg)
    except ValueError:
        raise ValueError('Non-numeric breakpoint number %s' % arg) from None
    try:
        bp = Breakpoint.bpbynumber[number]
    except IndexError:
        raise ValueError('Breakpoint number %d out of range' % number) from None
    if bp is None:
        raise ValueError('Breakpoint %d already deleted' % number)
    return bp