import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def is_copy_on_write(self):
    """
        @rtype:  bool
        @return: C{True} if all pages in this region are marked as
            copy-on-write. This means the pages are writeable, but changes
            are not propagated to disk.
        @note:
            Tipically data sections in executable images are marked like this.
        """
    return self.has_content() and bool(self.Protect & self.COPY_ON_WRITE)