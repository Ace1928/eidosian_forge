import fnmatch
import sys
import os
from inspect import CO_GENERATOR, CO_COROUTINE, CO_ASYNC_GENERATOR
def clear_break(self, filename, lineno):
    """Delete breakpoints for filename:lineno.

        If no breakpoints were set, return an error message.
        """
    filename = self.canonic(filename)
    if filename not in self.breaks:
        return 'There are no breakpoints in %s' % filename
    if lineno not in self.breaks[filename]:
        return 'There is no breakpoint at %s:%d' % (filename, lineno)
    for bp in Breakpoint.bplist[filename, lineno][:]:
        bp.deleteMe()
    self._prune_breaks(filename, lineno)
    return None