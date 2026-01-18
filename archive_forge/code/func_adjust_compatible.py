from contextlib import nullcontext
import matplotlib as mpl
from matplotlib._constrained_layout import do_constrained_layout
from matplotlib._tight_layout import (get_subplotspec_list,
@property
def adjust_compatible(self):
    """
        Return a boolean if the layout engine is compatible with
        `~.Figure.subplots_adjust`.
        """
    if self._adjust_compatible is None:
        raise NotImplementedError
    return self._adjust_compatible