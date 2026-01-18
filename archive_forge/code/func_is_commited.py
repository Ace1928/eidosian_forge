import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def is_commited(self):
    """
        @rtype:  bool
        @return: C{True} if the memory in this region is commited.
        """
    return self.State == MEM_COMMIT