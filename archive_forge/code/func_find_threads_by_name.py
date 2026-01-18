from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def find_threads_by_name(self, name, bExactMatch=True):
    """
        Find threads by name, using different search methods.

        @type  name: str, None
        @param name: Name to look for. Use C{None} to find nameless threads.

        @type  bExactMatch: bool
        @param bExactMatch: C{True} if the name must be
            B{exactly} as given, C{False} if the name can be
            loosely matched.

            This parameter is ignored when C{name} is C{None}.

        @rtype:  list( L{Thread} )
        @return: All threads matching the given name.
        """
    found_threads = list()
    if name is None:
        for aThread in self.iter_threads():
            if aThread.get_name() is None:
                found_threads.append(aThread)
    elif bExactMatch:
        for aThread in self.iter_threads():
            if aThread.get_name() == name:
                found_threads.append(aThread)
    else:
        for aThread in self.iter_threads():
            t_name = aThread.get_name()
            if t_name is not None and name in t_name:
                found_threads.append(aThread)
    return found_threads