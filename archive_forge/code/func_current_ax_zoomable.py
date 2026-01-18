import warnings
import itertools
from contextlib import contextmanager
from packaging.version import Version
import numpy as np
import matplotlib as mpl
from matplotlib import transforms
from .. import utils
@property
def current_ax_zoomable(self):
    return self.ax_zoomable(self._current_ax)