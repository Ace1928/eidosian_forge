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
def start_service(name, argv=None):
    """
        Start the service given by name.

        @warn: This method requires UAC elevation in Windows Vista and above.

        @see: L{stop_service}, L{pause_service}, L{resume_service}

        @type  name: str
        @param name: Service unique name. You can get this value from the
            C{ServiceName} member of the service descriptors returned by
            L{get_services} or L{get_active_services}.
        """
    with win32.OpenSCManager(dwDesiredAccess=win32.SC_MANAGER_CONNECT) as hSCManager:
        with win32.OpenService(hSCManager, name, dwDesiredAccess=win32.SERVICE_START) as hService:
            win32.StartService(hService)