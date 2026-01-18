from __future__ import annotations
import ctypes
import os
import sys
def get_base_class_and_magic_number(lib_file, seek=None):
    if seek is None:
        seek = lib_file.tell()
    else:
        lib_file.seek(seek)
    magic_number = ctypes.c_uint32.from_buffer_copy(lib_file.read(ctypes.sizeof(ctypes.c_uint32))).value
    if magic_number in [FAT_CIGAM, FAT_CIGAM_64, MH_CIGAM, MH_CIGAM_64]:
        if sys.byteorder == 'little':
            BaseClass = ctypes.BigEndianStructure
        else:
            BaseClass = ctypes.LittleEndianStructure
        magic_number = swap32(magic_number)
    else:
        BaseClass = ctypes.Structure
    lib_file.seek(seek)
    return (BaseClass, magic_number)