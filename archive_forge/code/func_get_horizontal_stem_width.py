from collections import namedtuple
import logging
import re
from ._mathtext_data import uni2type1
def get_horizontal_stem_width(self):
    """
        Return the standard horizontal stem width as float, or *None* if
        not specified in AFM file.
        """
    return self._header.get(b'StdHW', None)