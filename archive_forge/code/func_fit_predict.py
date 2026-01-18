import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn import preprocessing, decomposition
import scprep
from . import utils
def fit_predict(self, G, sample_indicator, likelihood=None, **kwargs):
    self.fit_transform(G, sample_indicator, likelihood, **kwargs)
    return self.predict()