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
@property
def s(self):
    if self._s is not None:
        if isinstance(self._s, numbers.Number):
            return self._s
        else:
            return self._s[self.plot_idx]
    else:
        return 200 / np.sqrt(self.size)