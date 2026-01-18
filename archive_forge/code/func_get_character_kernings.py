import os
import mmap
import struct
import codecs
def get_character_kernings(self):
    """Return a dictionary of (left,right)->kerning

        The key of the dictionary is a tuple of ``(left, right)``
        where each element is a unit-length unicode string.  The
        value of the dictionary is the horizontal pairwise kerning
        in em.
        """
    if not self._character_kernings:
        gmap = self.get_glyph_map()
        kerns = self.get_glyph_kernings()
        self._character_kernings = {}
        for pair, value in kerns.items():
            lglyph, rglyph = pair
            lchar = lglyph in gmap and gmap[lglyph] or None
            rchar = rglyph in gmap and gmap[rglyph] or None
            if lchar and rchar:
                self._character_kernings[lchar, rchar] = value
    return self._character_kernings