import fnmatch
import sys
import os
from inspect import CO_GENERATOR, CO_COROUTINE, CO_ASYNC_GENERATOR
@staticmethod
def clearBreakpoints():
    Breakpoint.next = 1
    Breakpoint.bplist = {}
    Breakpoint.bpbynumber = [None]