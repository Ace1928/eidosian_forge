from collections.abc import Mapping
import functools
import numpy as np
from numpy import ma
import matplotlib as mpl
from matplotlib import _api, colors, cbook, scale
from matplotlib._cm import datad
from matplotlib._cm_listed import cmaps as cmaps_listed
@_api.deprecated('3.7', alternative='``matplotlib.colormaps.register(name)``')
def register_cmap(name=None, cmap=None, *, override_builtin=False):
    """
    Add a colormap to the set recognized by :func:`get_cmap`.

    Register a new colormap to be accessed by name ::

        LinearSegmentedColormap('swirly', data, lut)
        register_cmap(cmap=swirly_cmap)

    Parameters
    ----------
    name : str, optional
       The name that can be used in :func:`get_cmap` or :rc:`image.cmap`

       If absent, the name will be the :attr:`~matplotlib.colors.Colormap.name`
       attribute of the *cmap*.

    cmap : matplotlib.colors.Colormap
       Despite being the second argument and having a default value, this
       is a required argument.

    override_builtin : bool

        Allow built-in colormaps to be overridden by a user-supplied
        colormap.

        Please do not use this unless you are sure you need it.
    """
    _api.check_isinstance((str, None), name=name)
    if name is None:
        try:
            name = cmap.name
        except AttributeError as err:
            raise ValueError('Arguments must include a name or a Colormap') from err
    _colormaps._allow_override_builtin = override_builtin
    _colormaps.register(cmap, name=name, force=override_builtin)
    _colormaps._allow_override_builtin = False