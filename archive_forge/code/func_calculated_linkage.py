import warnings
import matplotlib as mpl
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
from . import cm
from .axisgrid import Grid
from ._compat import get_colormap
from .utils import (
@property
def calculated_linkage(self):
    try:
        return self._calculate_linkage_fastcluster()
    except ImportError:
        if np.prod(self.shape) >= 10000:
            msg = 'Clustering large matrix with scipy. Installing `fastcluster` may give better performance.'
            warnings.warn(msg)
    return self._calculate_linkage_scipy()