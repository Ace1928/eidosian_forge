import inspect
import textwrap
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _docstring
from matplotlib.ticker import (
from matplotlib.transforms import Transform, IdentityTransform
def get_scale_names():
    """Return the names of the available scales."""
    return sorted(_scale_mapping)