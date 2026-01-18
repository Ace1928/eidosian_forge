import os
import mmap
import struct
import codecs
def get_glyph_kernings(self):
    """Return a dictionary of (left,right)->kerning

        The key of the dictionary is a tuple of ``(left, right)``
        where each element is a glyph index.  The value of the dictionary is
        the horizontal pairwise kerning in em.
        """
    if self._glyph_kernings:
        return self._glyph_kernings
    header = _read_kern_header_table(self._data, self._tables['kern'].offset)
    offset = self._tables['kern'].offset + header.size
    kernings = {}
    for i in range(header.n_tables):
        header = _read_kern_subtable_header(self._data, offset)
        if header.coverage & header.horizontal_mask and (not header.coverage & header.minimum_mask) and (not header.coverage & header.perpendicular_mask):
            if header.coverage & header.format_mask == 0:
                self._add_kernings_format0(kernings, offset + header.size)
        offset += header.length
    self._glyph_kernings = kernings
    return kernings