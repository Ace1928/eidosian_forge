from __future__ import with_statement
import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump, HexInput
from winappdbg.util import Regenerator, PathOperations, MemoryAddresses
from winappdbg.module import Module, _ModuleContainer
from winappdbg.thread import Thread, _ThreadContainer
from winappdbg.window import Window
from winappdbg.search import Search, \
from winappdbg.disasm import Disassembler
import re
import os
import os.path
import ctypes
import struct
import warnings
import traceback
def get_image_name(self):
    """
        @rtype:  int
        @return: Filename of the process main module.

            This method does it's best to retrieve the filename.
            However sometimes this is not possible, so C{None} may
            be returned instead.
        """
    mainModule = None
    try:
        mainModule = self.get_main_module()
        name = mainModule.fileName
        if not name:
            name = None
    except (KeyError, AttributeError, WindowsError):
        name = None
    if not name:
        try:
            hProcess = self.get_handle(win32.PROCESS_QUERY_LIMITED_INFORMATION)
            name = win32.QueryFullProcessImageName(hProcess)
        except (AttributeError, WindowsError):
            name = None
    if not name:
        try:
            hProcess = self.get_handle(win32.PROCESS_QUERY_INFORMATION)
            name = win32.GetProcessImageFileName(hProcess)
            if name:
                name = PathOperations.native_to_win32_pathname(name)
            else:
                name = None
        except (AttributeError, WindowsError):
            if not name:
                name = None
    if not name:
        try:
            hProcess = self.get_handle(win32.PROCESS_VM_READ | win32.PROCESS_QUERY_INFORMATION)
            try:
                name = win32.GetModuleFileNameEx(hProcess)
            except WindowsError:
                name = win32.GetModuleFileNameEx(hProcess, self.get_image_base())
            if name:
                name = PathOperations.native_to_win32_pathname(name)
            else:
                name = None
        except (AttributeError, WindowsError):
            if not name:
                name = None
    if not name:
        try:
            peb = self.get_peb()
            pp = self.read_structure(peb.ProcessParameters, win32.RTL_USER_PROCESS_PARAMETERS)
            s = pp.ImagePathName
            name = self.peek_string(s.Buffer, dwMaxSize=s.MaximumLength, fUnicode=True)
            if name:
                name = PathOperations.native_to_win32_pathname(name)
            else:
                name = None
        except (AttributeError, WindowsError):
            name = None
    if not name and mainModule is not None:
        try:
            name = mainModule.get_filename()
            if not name:
                name = None
        except (AttributeError, WindowsError):
            name = None
    if name and mainModule is not None:
        mainModule.fileName = name
    return name