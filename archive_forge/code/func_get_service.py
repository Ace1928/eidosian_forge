from __future__ import with_statement
from winappdbg import win32
from winappdbg.registry import Registry
from winappdbg.textio import HexInput, HexDump
from winappdbg.util import Regenerator, PathOperations, MemoryAddresses, DebugRegister, \
from winappdbg.process import _ProcessContainer
from winappdbg.window import Window
import sys
import os
import ctypes
import warnings
from os import path, getenv
@staticmethod
def get_service(name):
    """
        Get the service descriptor for the given service name.

        @see: L{start_service}, L{stop_service},
            L{pause_service}, L{resume_service}

        @type  name: str
        @param name: Service unique name. You can get this value from the
            C{ServiceName} member of the service descriptors returned by
            L{get_services} or L{get_active_services}.

        @rtype:  L{win32.ServiceStatusProcess}
        @return: Service status descriptor.
        """
    with win32.OpenSCManager(dwDesiredAccess=win32.SC_MANAGER_ENUMERATE_SERVICE) as hSCManager:
        with win32.OpenService(hSCManager, name, dwDesiredAccess=win32.SERVICE_QUERY_STATUS) as hService:
            try:
                return win32.QueryServiceStatusEx(hService)
            except AttributeError:
                return win32.QueryServiceStatus(hService)