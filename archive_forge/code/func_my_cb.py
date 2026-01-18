import re
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy.optimize import check_grad
from sklearn import clone
from sklearn.datasets import load_iris, make_blobs, make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state
def my_cb(transformation, n_iter):
    assert transformation.shape == (iris_data.shape[1] ** 2,)
    rem_iter = max_iter - n_iter
    print('{} iterations remaining...'.format(rem_iter))