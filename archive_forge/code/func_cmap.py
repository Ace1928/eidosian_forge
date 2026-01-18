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
def cmap(self):
    if self._cmap is not None:
        if isinstance(self._cmap, dict):
            return mpl.colors.ListedColormap([mpl.colors.to_rgba(self._cmap[label]) for label in self.labels])
        elif self.list_cmap():
            return create_colormap(self._cmap)
        elif isinstance(self._cmap, str):
            return self.process_string_cmap(self._cmap)
        else:
            return self._cmap
    elif self.constant_c() or self.array_c():
        return None
    elif self.discrete:
        return colors.tab(n=self.n_c_unique)
    else:
        return self.process_string_cmap('inferno')