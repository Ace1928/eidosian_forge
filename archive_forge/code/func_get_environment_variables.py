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
def get_environment_variables(self):
    """
        Retrieves the environment variables with wich the program is running.

        @rtype:  list of tuple(compat.unicode, compat.unicode)
        @return: Environment keys and values as found in the process memory.

        @raise WindowsError: On error an exception is raised.
        """
    data = self.peek(*self.get_environment_block())
    tmp = ctypes.create_string_buffer(data)
    buffer = ctypes.create_unicode_buffer(len(data))
    ctypes.memmove(buffer, tmp, len(data))
    del tmp
    pos = 0
    while buffer[pos] != u'\x00':
        pos += 1
    pos += 1
    environment = []
    while buffer[pos] != u'\x00':
        env_name_pos = pos
        env_name = u''
        found_name = False
        while buffer[pos] != u'\x00':
            char = buffer[pos]
            if char == u'=':
                if env_name_pos == pos:
                    env_name_pos += 1
                    pos += 1
                    continue
                pos += 1
                found_name = True
                break
            env_name += char
            pos += 1
        if not found_name:
            break
        env_value = u''
        while buffer[pos] != u'\x00':
            env_value += buffer[pos]
            pos += 1
        pos += 1
        environment.append((env_name, env_value))
    if environment:
        environment.pop()
    return environment