import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
def set_verts(self, verts, closed=True):
    """
        Set the vertices of the polygons.

        Parameters
        ----------
        verts : list of array-like
            The sequence of polygons [*verts0*, *verts1*, ...] where each
            element *verts_i* defines the vertices of polygon *i* as a 2D
            array-like of shape (M, 2).
        closed : bool, default: True
            Whether the polygon should be closed by adding a CLOSEPOLY
            connection at the end.
        """
    self.stale = True
    if isinstance(verts, np.ma.MaskedArray):
        verts = verts.astype(float).filled(np.nan)
    if not closed:
        self._paths = [mpath.Path(xy) for xy in verts]
        return
    if isinstance(verts, np.ndarray) and len(verts.shape) == 3:
        verts_pad = np.concatenate((verts, verts[:, :1]), axis=1)
        codes = np.empty(verts_pad.shape[1], dtype=mpath.Path.code_type)
        codes[:] = mpath.Path.LINETO
        codes[0] = mpath.Path.MOVETO
        codes[-1] = mpath.Path.CLOSEPOLY
        self._paths = [mpath.Path(xy, codes) for xy in verts_pad]
        return
    self._paths = []
    for xy in verts:
        if len(xy):
            self._paths.append(mpath.Path._create_closed(xy))
        else:
            self._paths.append(mpath.Path(xy))