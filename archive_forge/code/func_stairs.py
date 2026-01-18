import functools
import itertools
import logging
import math
from numbers import Integral, Number, Real
import numpy as np
from numpy import ma
import matplotlib as mpl
import matplotlib.category  # Register category unit converter as side effect.
import matplotlib.cbook as cbook
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.contour as mcontour
import matplotlib.dates  # noqa # Register date unit converter as side effect.
import matplotlib.image as mimage
import matplotlib.legend as mlegend
import matplotlib.lines as mlines
import matplotlib.markers as mmarkers
import matplotlib.mlab as mlab
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.quiver as mquiver
import matplotlib.stackplot as mstack
import matplotlib.streamplot as mstream
import matplotlib.table as mtable
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import matplotlib.tri as mtri
import matplotlib.units as munits
from matplotlib import _api, _docstring, _preprocess_data
from matplotlib.axes._base import (
from matplotlib.axes._secondary_axes import SecondaryAxis
from matplotlib.container import BarContainer, ErrorbarContainer, StemContainer
@_preprocess_data()
def stairs(self, values, edges=None, *, orientation='vertical', baseline=0, fill=False, **kwargs):
    """
        A stepwise constant function as a line with bounding edges
        or a filled plot.

        Parameters
        ----------
        values : array-like
            The step heights.

        edges : array-like
            The edge positions, with ``len(edges) == len(vals) + 1``,
            between which the curve takes on vals values.

        orientation : {'vertical', 'horizontal'}, default: 'vertical'
            The direction of the steps. Vertical means that *values* are along
            the y-axis, and edges are along the x-axis.

        baseline : float, array-like or None, default: 0
            The bottom value of the bounding edges or when
            ``fill=True``, position of lower edge. If *fill* is
            True or an array is passed to *baseline*, a closed
            path is drawn.

        fill : bool, default: False
            Whether the area under the step curve should be filled.

        Returns
        -------
        StepPatch : `~matplotlib.patches.StepPatch`

        Other Parameters
        ----------------
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER

        **kwargs
            `~matplotlib.patches.StepPatch` properties

        """
    if 'color' in kwargs:
        _color = kwargs.pop('color')
    else:
        _color = self._get_lines.get_next_color()
    if fill:
        kwargs.setdefault('linewidth', 0)
        kwargs.setdefault('facecolor', _color)
    else:
        kwargs.setdefault('edgecolor', _color)
    if edges is None:
        edges = np.arange(len(values) + 1)
    edges, values, baseline = self._process_unit_info([('x', edges), ('y', values), ('y', baseline)], kwargs)
    patch = mpatches.StepPatch(values, edges, baseline=baseline, orientation=orientation, fill=fill, **kwargs)
    self.add_patch(patch)
    if baseline is None:
        baseline = 0
    if orientation == 'vertical':
        patch.sticky_edges.y.append(np.min(baseline))
        self.update_datalim([(edges[0], np.min(baseline))])
    else:
        patch.sticky_edges.x.append(np.min(baseline))
        self.update_datalim([(np.min(baseline), edges[0])])
    self._request_autoscale_view()
    return patch