from contextlib import nullcontext
import matplotlib as mpl
from matplotlib._constrained_layout import do_constrained_layout
from matplotlib._tight_layout import (get_subplotspec_list,
@property
def colorbar_gridspec(self):
    """
        Return a boolean if the layout engine creates colorbars using a
        gridspec.
        """
    if self._colorbar_gridspec is None:
        raise NotImplementedError
    return self._colorbar_gridspec