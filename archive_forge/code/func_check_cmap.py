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
def check_cmap(self):
    if isinstance(self._cmap, dict):
        if self.constant_c() or self.array_c():
            raise ValueError('Expected list-like `c` with dictionary cmap. Got {}'.format(type(self._c)))
        elif not self.discrete:
            raise ValueError('Cannot use dictionary cmap with continuous data.')
        elif np.any([color not in self._cmap for color in np.unique(self._c)]):
            missing = set(np.unique(self._c).tolist()).difference(self._cmap.keys())
            raise ValueError('Dictionary cmap requires a color for every unique entry in `c`. Missing colors for [{}]'.format(', '.join([str(color) for color in missing])))
    elif self.list_cmap():
        if self.constant_c() or self.array_c():
            raise ValueError('Expected list-like `c` with list cmap. Got {}'.format(type(self._c)))