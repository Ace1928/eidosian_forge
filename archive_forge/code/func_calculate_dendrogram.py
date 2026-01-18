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
def calculate_dendrogram(self):
    """Calculates a dendrogram based on the linkage matrix

        Made a separate function, not a property because don't want to
        recalculate the dendrogram every time it is accessed.

        Returns
        -------
        dendrogram : dict
            Dendrogram dictionary as returned by scipy.cluster.hierarchy
            .dendrogram. The important key-value pairing is
            "reordered_ind" which indicates the re-ordering of the matrix
        """
    return hierarchy.dendrogram(self.linkage, no_plot=True, color_threshold=-np.inf)