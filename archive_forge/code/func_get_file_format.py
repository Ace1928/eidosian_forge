import copy
import ctypes
import ctypes.util
import os
import sys
from .exceptions import DecodeError
from .base import AudioFile
def get_file_format(self):
    """Get the file format description. This describes the type of
        data stored on disk.
        """
    if self._file_fmt is not None:
        return self._file_fmt
    desc = AudioStreamBasicDescription()
    size = ctypes.c_int(ctypes.sizeof(desc))
    check(_coreaudio.ExtAudioFileGetProperty(self._obj, PROP_FILE_DATA_FORMAT, ctypes.byref(size), ctypes.byref(desc)))
    self._file_fmt = desc
    return desc