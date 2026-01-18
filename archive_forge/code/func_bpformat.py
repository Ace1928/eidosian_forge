import fnmatch
import sys
import os
from inspect import CO_GENERATOR, CO_COROUTINE, CO_ASYNC_GENERATOR
def bpformat(self):
    """Return a string with information about the breakpoint.

        The information includes the breakpoint number, temporary
        status, file:line position, break condition, number of times to
        ignore, and number of times hit.

        """
    if self.temporary:
        disp = 'del  '
    else:
        disp = 'keep '
    if self.enabled:
        disp = disp + 'yes  '
    else:
        disp = disp + 'no   '
    ret = '%-4dbreakpoint   %s at %s:%d' % (self.number, disp, self.file, self.line)
    if self.cond:
        ret += '\n\tstop only if %s' % (self.cond,)
    if self.ignore:
        ret += '\n\tignore next %d hits' % (self.ignore,)
    if self.hits:
        if self.hits > 1:
            ss = 's'
        else:
            ss = ''
        ret += '\n\tbreakpoint already hit %d time%s' % (self.hits, ss)
    return ret