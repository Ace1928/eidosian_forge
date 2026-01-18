import warnings
import itertools
from contextlib import contextmanager
from packaging.version import Version
import numpy as np
import matplotlib as mpl
from matplotlib import transforms
from .. import utils
@staticmethod
def ax_zoomable(ax):
    return bool(ax and ax.get_navigate())