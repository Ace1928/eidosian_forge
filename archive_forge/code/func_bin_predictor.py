import copy
from textwrap import dedent
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from . import utils
from . import algorithms as algo
from .axisgrid import FacetGrid, _facet_docs
def bin_predictor(self, bins):
    """Discretize a predictor by assigning value to closest bin."""
    x = np.asarray(self.x)
    if np.isscalar(bins):
        percentiles = np.linspace(0, 100, bins + 2)[1:-1]
        bins = np.percentile(x, percentiles)
    else:
        bins = np.ravel(bins)
    dist = np.abs(np.subtract.outer(x, bins))
    x_binned = bins[np.argmin(dist, axis=1)].ravel()
    return (x_binned, bins)