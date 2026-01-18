import numpy as np
import matplotlib as mpl
from matplotlib import _api, artist as martist
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Bbox
def get_subplotspec_list(axes_list, grid_spec=None):
    """
    Return a list of subplotspec from the given list of axes.

    For an instance of axes that does not support subplotspec, None is inserted
    in the list.

    If grid_spec is given, None is inserted for those not from the given
    grid_spec.
    """
    subplotspec_list = []
    for ax in axes_list:
        axes_or_locator = ax.get_axes_locator()
        if axes_or_locator is None:
            axes_or_locator = ax
        if hasattr(axes_or_locator, 'get_subplotspec'):
            subplotspec = axes_or_locator.get_subplotspec()
            if subplotspec is not None:
                subplotspec = subplotspec.get_topmost_subplotspec()
                gs = subplotspec.get_gridspec()
                if grid_spec is not None:
                    if gs != grid_spec:
                        subplotspec = None
                elif gs.locally_modified_subplot_params():
                    subplotspec = None
        else:
            subplotspec = None
        subplotspec_list.append(subplotspec)
    return subplotspec_list