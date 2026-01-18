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
@property
def estimate_data(self):
    """Data with a point estimate and CI for each discrete x value."""
    x, y = (self.x_discrete, self.y)
    vals = sorted(np.unique(x))
    points, cis = ([], [])
    for val in vals:
        _y = y[x == val]
        est = self.x_estimator(_y)
        points.append(est)
        if self.x_ci is None:
            cis.append(None)
        else:
            units = None
            if self.x_ci == 'sd':
                sd = np.std(_y)
                _ci = (est - sd, est + sd)
            else:
                if self.units is not None:
                    units = self.units[x == val]
                boots = algo.bootstrap(_y, func=self.x_estimator, n_boot=self.n_boot, units=units, seed=self.seed)
                _ci = utils.ci(boots, self.x_ci)
            cis.append(_ci)
    return (vals, points, cis)