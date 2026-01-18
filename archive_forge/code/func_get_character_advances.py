import os
import mmap
import struct
import codecs
def get_character_advances(self):
    """Return a dictionary of character->advance.

        They key of the dictionary is a unit-length unicode string,
        and the value is a float giving the horizontal advance in
        em.
        """
    if self._character_advances:
        return self._character_advances
    ga = self.get_glyph_advances()
    gmap = self.get_glyph_map()
    self._character_advances = {}
    for i in range(len(ga)):
        if i in gmap and (not gmap[i] in self._character_advances):
            self._character_advances[gmap[i]] = ga[i]
    return self._character_advances