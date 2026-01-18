import os
import mmap
import struct
import codecs
def get_glyph_map(self):
    """Calculate and return a reverse character map.

        Returns a dictionary where the key is a glyph index and the
        value is a unit-length unicode string.
        """
    if self._glyph_map:
        return self._glyph_map
    cmap = self.get_character_map()
    self._glyph_map = {}
    for ch, glyph in cmap.items():
        if not glyph in self._glyph_map:
            self._glyph_map[glyph] = ch
    return self._glyph_map