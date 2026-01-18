import fnmatch
import sys
import os
from inspect import CO_GENERATOR, CO_COROUTINE, CO_ASYNC_GENERATOR
def break_here(self, frame):
    """Return True if there is an effective breakpoint for this line.

        Check for line or function breakpoint and if in effect.
        Delete temporary breakpoints if effective() says to.
        """
    filename = self.canonic(frame.f_code.co_filename)
    if filename not in self.breaks:
        return False
    lineno = frame.f_lineno
    if lineno not in self.breaks[filename]:
        lineno = frame.f_code.co_firstlineno
        if lineno not in self.breaks[filename]:
            return False
    bp, flag = effective(filename, lineno, frame)
    if bp:
        self.currentbp = bp.number
        if flag and bp.temporary:
            self.do_clear(str(bp.number))
        return True
    else:
        return False