import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def is_image(self):
    """
        @rtype:  bool
        @return: C{True} if the memory in this region belongs to an executable
            image.
        """
    return self.Type == MEM_IMAGE