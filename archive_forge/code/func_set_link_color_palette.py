import warnings
import bisect
from collections import deque
import numpy as np
from . import _hierarchy, _optimal_leaf_ordering
import scipy.spatial.distance as distance
from scipy._lib._array_api import array_namespace, as_xparray, copy
from scipy._lib._disjoint_set import DisjointSet
def set_link_color_palette(palette):
    """
    Set list of matplotlib color codes for use by dendrogram.

    Note that this palette is global (i.e., setting it once changes the colors
    for all subsequent calls to `dendrogram`) and that it affects only the
    the colors below ``color_threshold``.

    Note that `dendrogram` also accepts a custom coloring function through its
    ``link_color_func`` keyword, which is more flexible and non-global.

    Parameters
    ----------
    palette : list of str or None
        A list of matplotlib color codes.  The order of the color codes is the
        order in which the colors are cycled through when color thresholding in
        the dendrogram.

        If ``None``, resets the palette to its default (which are matplotlib
        default colors C1 to C9).

    Returns
    -------
    None

    See Also
    --------
    dendrogram

    Notes
    -----
    Ability to reset the palette with ``None`` added in SciPy 0.17.0.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.cluster import hierarchy
    >>> ytdist = np.array([662., 877., 255., 412., 996., 295., 468., 268.,
    ...                    400., 754., 564., 138., 219., 869., 669.])
    >>> Z = hierarchy.linkage(ytdist, 'single')
    >>> dn = hierarchy.dendrogram(Z, no_plot=True)
    >>> dn['color_list']
    ['C1', 'C0', 'C0', 'C0', 'C0']
    >>> hierarchy.set_link_color_palette(['c', 'm', 'y', 'k'])
    >>> dn = hierarchy.dendrogram(Z, no_plot=True, above_threshold_color='b')
    >>> dn['color_list']
    ['c', 'b', 'b', 'b', 'b']
    >>> dn = hierarchy.dendrogram(Z, no_plot=True, color_threshold=267,
    ...                           above_threshold_color='k')
    >>> dn['color_list']
    ['c', 'm', 'm', 'k', 'k']

    Now, reset the color palette to its default:

    >>> hierarchy.set_link_color_palette(None)

    """
    if palette is None:
        palette = _link_line_colors_default
    elif not isinstance(palette, (list, tuple)):
        raise TypeError('palette must be a list or tuple')
    _ptypes = [isinstance(p, str) for p in palette]
    if False in _ptypes:
        raise TypeError('all palette list elements must be color strings')
    global _link_line_colors
    _link_line_colors = palette