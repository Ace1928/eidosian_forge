import ctypes
import os_win.conf
from os_win.utils.winapi import wintypes
def get_target_mapping_struct(entry_count=0):

    class HBA_FCPTargetMapping(ctypes.Structure):
        _fields_ = [('NumberOfEntries', ctypes.c_uint32), ('Entries', HBA_FCPScsiEntry * entry_count)]

        def __init__(self, entry_count):
            self.NumberOfEntries = entry_count
            self.Entries = (HBA_FCPScsiEntry * entry_count)()
    return HBA_FCPTargetMapping(entry_count)