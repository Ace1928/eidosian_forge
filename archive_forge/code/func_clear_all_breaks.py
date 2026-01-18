import fnmatch
import sys
import os
from inspect import CO_GENERATOR, CO_COROUTINE, CO_ASYNC_GENERATOR
def clear_all_breaks(self):
    """Delete all existing breakpoints.

        If none were set, return an error message.
        """
    if not self.breaks:
        return 'There are no breakpoints'
    for bp in Breakpoint.bpbynumber:
        if bp:
            bp.deleteMe()
    self.breaks = {}
    return None