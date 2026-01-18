import fnmatch
import sys
import os
from inspect import CO_GENERATOR, CO_COROUTINE, CO_ASYNC_GENERATOR
def runeval(self, expr, globals=None, locals=None):
    """Debug an expression executed via the eval() function.

        globals defaults to __main__.dict; locals defaults to globals.
        """
    if globals is None:
        import __main__
        globals = __main__.__dict__
    if locals is None:
        locals = globals
    self.reset()
    sys.settrace(self.trace_dispatch)
    try:
        return eval(expr, globals, locals)
    except BdbQuit:
        pass
    finally:
        self.quitting = True
        sys.settrace(None)