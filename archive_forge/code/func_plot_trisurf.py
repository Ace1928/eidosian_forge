from collections import defaultdict
import functools
import itertools
import math
import textwrap
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, _docstring, _preprocess_data
import matplotlib.artist as martist
import matplotlib.axes as maxes
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.image as mimage
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.container as mcontainer
import matplotlib.transforms as mtransforms
from matplotlib.axes import Axes
from matplotlib.axes._base import _axis_method_wrapper, _process_plot_format
from matplotlib.transforms import Bbox
from matplotlib.tri._triangulation import Triangulation
from . import art3d
from . import proj3d
from . import axis3d
def plot_trisurf(self, *args, color=None, norm=None, vmin=None, vmax=None, lightsource=None, **kwargs):
    """
        Plot a triangulated surface.

        The (optional) triangulation can be specified in one of two ways;
        either::

          plot_trisurf(triangulation, ...)

        where triangulation is a `~matplotlib.tri.Triangulation` object, or::

          plot_trisurf(X, Y, ...)
          plot_trisurf(X, Y, triangles, ...)
          plot_trisurf(X, Y, triangles=triangles, ...)

        in which case a Triangulation object will be created.  See
        `.Triangulation` for an explanation of these possibilities.

        The remaining arguments are::

          plot_trisurf(..., Z)

        where *Z* is the array of values to contour, one per point
        in the triangulation.

        Parameters
        ----------
        X, Y, Z : array-like
            Data values as 1D arrays.
        color
            Color of the surface patches.
        cmap
            A colormap for the surface patches.
        norm : Normalize
            An instance of Normalize to map values to colors.
        vmin, vmax : float, default: None
            Minimum and maximum value to map.
        shade : bool, default: True
            Whether to shade the facecolors.  Shading is always disabled when
            *cmap* is specified.
        lightsource : `~matplotlib.colors.LightSource`
            The lightsource to use when *shade* is True.
        **kwargs
            All other keyword arguments are passed on to
            :class:`~mpl_toolkits.mplot3d.art3d.Poly3DCollection`

        Examples
        --------
        .. plot:: gallery/mplot3d/trisurf3d.py
        .. plot:: gallery/mplot3d/trisurf3d_2.py
        """
    had_data = self.has_data()
    if color is None:
        color = self._get_lines.get_next_color()
    color = np.array(mcolors.to_rgba(color))
    cmap = kwargs.get('cmap', None)
    shade = kwargs.pop('shade', cmap is None)
    tri, args, kwargs = Triangulation.get_from_args_and_kwargs(*args, **kwargs)
    try:
        z = kwargs.pop('Z')
    except KeyError:
        z, *args = args
    z = np.asarray(z)
    triangles = tri.get_masked_triangles()
    xt = tri.x[triangles]
    yt = tri.y[triangles]
    zt = z[triangles]
    verts = np.stack((xt, yt, zt), axis=-1)
    if cmap:
        polyc = art3d.Poly3DCollection(verts, *args, **kwargs)
        avg_z = verts[:, :, 2].mean(axis=1)
        polyc.set_array(avg_z)
        if vmin is not None or vmax is not None:
            polyc.set_clim(vmin, vmax)
        if norm is not None:
            polyc.set_norm(norm)
    else:
        polyc = art3d.Poly3DCollection(verts, *args, shade=shade, lightsource=lightsource, facecolors=color, **kwargs)
    self.add_collection(polyc)
    self.auto_scale_xyz(tri.x, tri.y, z, had_data)
    return polyc