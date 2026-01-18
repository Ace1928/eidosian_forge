import fnmatch
import sys
import os
from inspect import CO_GENERATOR, CO_COROUTINE, CO_ASYNC_GENERATOR
def deleteMe(self):
    """Delete the breakpoint from the list associated to a file:line.

        If it is the last breakpoint in that position, it also deletes
        the entry for the file:line.
        """
    index = (self.file, self.line)
    self.bpbynumber[self.number] = None
    self.bplist[index].remove(self)
    if not self.bplist[index]:
        del self.bplist[index]