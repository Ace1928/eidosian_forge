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
def constant_c(self):
    """Check if ``c`` is constant.

        Returns
        -------
        c : ``str`` or ``None``
            Either None or a single matplotlib color
        """
    if self._c is None or isinstance(self._c, str):
        return True
    elif hasattr(self._c, '__len__') and len(self._c) == self.size:
        return False
    else:
        return mpl.colors.is_color_like(self._c)