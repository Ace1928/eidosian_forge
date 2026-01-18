import fnmatch
import sys
import os
from inspect import CO_GENERATOR, CO_COROUTINE, CO_ASYNC_GENERATOR
def effective(file, line, frame):
    """Return (active breakpoint, delete temporary flag) or (None, None) as
       breakpoint to act upon.

       The "active breakpoint" is the first entry in bplist[line, file] (which
       must exist) that is enabled, for which checkfuncname is True, and that
       has neither a False condition nor a positive ignore count.  The flag,
       meaning that a temporary breakpoint should be deleted, is False only
       when the condiion cannot be evaluated (in which case, ignore count is
       ignored).

       If no such entry exists, then (None, None) is returned.
    """
    possibles = Breakpoint.bplist[file, line]
    for b in possibles:
        if not b.enabled:
            continue
        if not checkfuncname(b, frame):
            continue
        b.hits += 1
        if not b.cond:
            if b.ignore > 0:
                b.ignore -= 1
                continue
            else:
                return (b, True)
        else:
            try:
                val = eval(b.cond, frame.f_globals, frame.f_locals)
                if val:
                    if b.ignore > 0:
                        b.ignore -= 1
                    else:
                        return (b, True)
            except:
                return (b, False)
    return (None, None)