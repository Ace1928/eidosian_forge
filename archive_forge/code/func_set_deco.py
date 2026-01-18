from __future__ import division
import sys
import unicodedata
from functools import reduce
def set_deco(self, deco):
    """Set the table decoration

        - 'deco' can be a combination of:

            Texttable.BORDER: Border around the table
            Texttable.HEADER: Horizontal line below the header
            Texttable.HLINES: Horizontal lines between rows
            Texttable.VLINES: Vertical lines between columns

           All of them are enabled by default

        - example:

            Texttable.BORDER | Texttable.HEADER
        """
    self._deco = deco
    self._hline_string = None
    return self