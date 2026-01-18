import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def get_chars(self):
    """
        This generator function is used to return all unicode character
        codes in the current charmap of a given face. For each character it
        also returns the corresponding glyph index.

        :return: character code, glyph index

        **Note**:
          Note that 'agindex' is set to 0 if the charmap is empty. The
          character code itself can be 0 in two cases: if the charmap is empty
          or if the value 0 is the first valid character code.
        """
    charcode, agindex = self.get_first_char()
    yield (charcode, agindex)
    while agindex != 0:
        charcode, agindex = self.get_next_char(charcode, 0)
        yield (charcode, agindex)