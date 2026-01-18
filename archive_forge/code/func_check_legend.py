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
def check_legend(self):
    if self._colorbar is not None:
        if self._legend is not None and self._legend != self._colorbar:
            raise ValueError('Received conflicting values for synonyms `legend={}` and `colorbar={}`'.format(self._legend, self._colorbar))
        else:
            self._legend = self._colorbar
    if self._legend:
        if self.array_c():
            warnings.warn('`c` is a color array and cannot be used to create a legend. To interpret these values as labels instead, provide a `cmap` dictionary with label-color pairs.', UserWarning)
            self._legend = False
        elif self.constant_c():
            warnings.warn('Cannot create a legend with constant `c={}`'.format(self.c), UserWarning)
            self._legend = False