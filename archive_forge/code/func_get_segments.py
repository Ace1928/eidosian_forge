import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
def get_segments(self):
    """
        Returns
        -------
        list
            List of segments in the LineCollection. Each list item contains an
            array of vertices.
        """
    segments = []
    for path in self._paths:
        vertices = [vertex for vertex, _ in path.iter_segments(simplify=False)]
        vertices = np.asarray(vertices)
        segments.append(vertices)
    return segments