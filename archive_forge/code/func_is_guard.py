import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def is_guard(self):
    """
        @rtype:  bool
        @return: C{True} if all pages in this region are guard pages.
        """
    return self.is_commited() and bool(self.Protect & PAGE_GUARD)