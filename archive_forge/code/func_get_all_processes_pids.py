import os
from ctypes import (
from ctypes.wintypes import DWORD, LONG
def get_all_processes_pids():
    """Return a dictionary with all processes pids as keys and their
       parents as value. Ignore processes with no parents.
    """
    h = CreateToolhelp32Snapshot()
    parents = {}
    pe = Process32First(h)
    while pe:
        if pe.th32ParentProcessID:
            parents[pe.th32ProcessID] = pe.th32ParentProcessID
        pe = Process32Next(h, pe)
    return parents