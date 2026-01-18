from . import utils
from scipy import sparse
from sklearn import decomposition
from sklearn import random_projection
import numpy as np
import pandas as pd
import sklearn.base
import warnings
@property
def explained_variance_ratio_(self):
    """Percentage of variance explained by each of the selected components.

        The sum of the ratios is equal to 1.0.
        If n_components is `None` then the number of components grows as`eps`
        gets smaller.
        """
    return self.pca_op.explained_variance_ratio_