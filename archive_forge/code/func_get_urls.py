import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
def get_urls(self):
    """
        Return a list of URLs, one for each element of the collection.

        The list contains *None* for elements without a URL. See
        :doc:`/gallery/misc/hyperlinks_sgskip` for an example.
        """
    return self._urls