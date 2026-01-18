import os
import mmap
import struct
import codecs
def get_font_selection_flags(self):
    """Return the font selection flags, as defined in OS/2 table"""
    if not self._font_selection_flags:
        OS2_table = _read_OS2_table(self._data, self._tables['OS/2'].offset)
        self._font_selection_flags = OS2_table.fs_selection
    return self._font_selection_flags