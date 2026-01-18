import numpy as np
from . import _api, _docstring
from .artist import Artist, allow_rasterization
from .patches import Rectangle
from .text import Text
from .transforms import Bbox
from .path import Path
def get_celld(self):
    """
        Return a dict of cells in the table mapping *(row, column)* to
        `.Cell`\\s.

        Notes
        -----
        You can also directly index into the Table object to access individual
        cells::

            cell = table[row, col]

        """
    return self._cells