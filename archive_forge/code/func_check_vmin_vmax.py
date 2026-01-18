from .. import select
from .. import utils
from .._lazyload import matplotlib as mpl
from . import colors
from .tools import create_colormap
from .tools import create_normalize
from .tools import generate_colorbar
from .tools import generate_legend
from .tools import label_axis
from .utils import _get_figure
from .utils import _in_ipynb
from .utils import _is_color_array
from .utils import _with_default
from .utils import parse_fontsize
from .utils import show
from .utils import temp_fontsize
import numbers
import numpy as np
import pandas as pd
import warnings
def check_vmin_vmax(self):
    if self.constant_c():
        if self._vmin_set is not None or self._vmax_set is not None:
            warnings.warn('Cannot set `vmin` or `vmax` with constant `c={}`. Setting `vmin = vmax = None`.'.format(self.c), UserWarning)
        self._vmin_set = None
        self._vmax_set = None
    elif self.discrete:
        if self._vmin_set is not None or self._vmax_set is not None:
            warnings.warn('Cannot set `vmin` or `vmax` with discrete data. Setting to `None`.', UserWarning)
        self._vmin_set = None
        self._vmax_set = None