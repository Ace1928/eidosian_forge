from .. import utils
from .._lazyload import matplotlib as mpl
from .._lazyload import mpl_toolkits
import numpy as np
import platform
@utils._with_pkg(pkg='matplotlib', min_version=3)
def shift_ticklabels(axis, dx=0, dy=0):
    """Shifts ticklabels on an axis.

    Parameters
    ----------
    axis : matplotlib.axis.{X,Y}Axis, mpl_toolkits.mplot3d.axis3d.{X,Y,Z}Axis
        Axis on which to draw labels and ticks
    dx : float, optional (default: 0)
        Horizontal shift
    dy : float, optional (default: 0)
    """
    offset = mpl.transforms.ScaledTranslation(dx, dy, axis.get_figure().dpi_scale_trans)
    for label in axis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)