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
def c_discrete(self):
    """Discretize ``c``.

        If c is discrete then this converts it to
        integers from 0 to `n_c_unique`
        """
    if self._c_discrete is None:
        if isinstance(self._cmap, dict):
            self._labels = np.array([k for k in self._cmap.keys() if k in self.c_unique])
            self._c_discrete = np.zeros_like(self._c, dtype=int)
            for i, label in enumerate(self._labels):
                self._c_discrete[self._c == label] = i
        else:
            self._c_discrete = np.zeros_like(self._c, dtype=int)
            self._c_discrete[self._mask], self._labels = pd.factorize(self._c_masked, sort=True)
    return self._c_discrete