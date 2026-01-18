import glob
import os
import struct
from .lazy_import import lazy_import
import ctypes
from breezy import cmdline
from breezy.i18n import gettext
def debug_memory_win32api(message='', short=True):
    """Use trace.note() to dump the running memory info."""
    from breezy import trace

    class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        """Used by GetProcessMemoryInfo"""
        _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong), ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t), ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t), ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t), ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t), ('PrivateUsage', ctypes.c_size_t)]
    cur_process = ctypes.windll.kernel32.GetCurrentProcess()
    mem_struct = PROCESS_MEMORY_COUNTERS_EX()
    ret = ctypes.windll.psapi.GetProcessMemoryInfo(cur_process, ctypes.byref(mem_struct), ctypes.sizeof(mem_struct))
    if not ret:
        trace.note(gettext('Failed to GetProcessMemoryInfo()'))
        return
    info = {'PageFaultCount': mem_struct.PageFaultCount, 'PeakWorkingSetSize': mem_struct.PeakWorkingSetSize, 'WorkingSetSize': mem_struct.WorkingSetSize, 'QuotaPeakPagedPoolUsage': mem_struct.QuotaPeakPagedPoolUsage, 'QuotaPagedPoolUsage': mem_struct.QuotaPagedPoolUsage, 'QuotaPeakNonPagedPoolUsage': mem_struct.QuotaPeakNonPagedPoolUsage, 'QuotaNonPagedPoolUsage': mem_struct.QuotaNonPagedPoolUsage, 'PagefileUsage': mem_struct.PagefileUsage, 'PeakPagefileUsage': mem_struct.PeakPagefileUsage, 'PrivateUsage': mem_struct.PrivateUsage}
    if short:
        trace.note(gettext('WorkingSize {0:>7}KiB\tPeakWorking {1:>7}KiB\t{2}').format(info['WorkingSetSize'] / 1024, info['PeakWorkingSetSize'] / 1024, message))
        return
    if message:
        trace.note('%s', message)
    trace.note(gettext('WorkingSize       %8d KiB'), info['WorkingSetSize'] / 1024)
    trace.note(gettext('PeakWorking       %8d KiB'), info['PeakWorkingSetSize'] / 1024)
    trace.note(gettext('PagefileUsage     %8d KiB'), info.get('PagefileUsage', 0) / 1024)
    trace.note(gettext('PeakPagefileUsage %8d KiB'), info.get('PeakPagefileUsage', 0) / 1024)
    trace.note(gettext('PrivateUsage      %8d KiB'), info.get('PrivateUsage', 0) / 1024)
    trace.note(gettext('PageFaultCount    %8d'), info.get('PageFaultCount', 0))