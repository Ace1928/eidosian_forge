import os
import mmap
import struct
import codecs
def is_bold(self):
    """Returns True iff the font describes itself as bold."""
    return bool(self.get_font_selection_flags() & 32)