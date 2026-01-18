import fnmatch
import sys
import os
from inspect import CO_GENERATOR, CO_COROUTINE, CO_ASYNC_GENERATOR
def clear_all_file_breaks(self, filename):
    """Delete all breakpoints in filename.

        If none were set, return an error message.
        """
    filename = self.canonic(filename)
    if filename not in self.breaks:
        return 'There are no breakpoints in %s' % filename
    for line in self.breaks[filename]:
        blist = Breakpoint.bplist[filename, line]
        for bp in blist:
            bp.deleteMe()
    del self.breaks[filename]
    return None