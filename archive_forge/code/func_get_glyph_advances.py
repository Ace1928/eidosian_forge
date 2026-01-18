import os
import mmap
import struct
import codecs
def get_glyph_advances(self):
    """Return a dictionary of glyph->advance.

        They key of the dictionary is the glyph index and the value is a float
        giving the horizontal advance in em.
        """
    hm = self.get_horizontal_metrics()
    return [float(m.advance_width) / self.header.units_per_em for m in hm]