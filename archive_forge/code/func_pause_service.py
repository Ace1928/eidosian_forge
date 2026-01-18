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
def pause_service(name):
    """
        Pause the service given by name.

        @warn: This method requires UAC elevation in Windows Vista and above.

        @note: Not all services support this.

        @see: L{get_services}, L{get_active_services},
            L{start_service}, L{stop_service}, L{resume_service}
        """
    with win32.OpenSCManager(dwDesiredAccess=win32.SC_MANAGER_CONNECT) as hSCManager:
        with win32.OpenService(hSCManager, name, dwDesiredAccess=win32.SERVICE_PAUSE_CONTINUE) as hService:
            win32.ControlService(hService, win32.SERVICE_CONTROL_PAUSE)