from .. import utils
from .._lazyload import matplotlib as mpl
from .._lazyload import mpl_toolkits
import numpy as np
import platform
def parse_fontsize(size=None, default=None):
    """Parse the user's input font size.

    Returns `size` if explicitly set by user,
    `default` if not set by user and the user's matplotlibrc is also default,
    or `None` otherwise (falling back to mpl defaults)

    Parameters
    ----------
    size
        Fontsize explicitly set by user
    default
        Desired default font size in
        xx-small, x-small, small, medium, large, x-large, xx-large,
        larger, smaller
    """
    if size is not None:
        return size
    elif _is_default_matplotlibrc():
        return default
    else:
        return None