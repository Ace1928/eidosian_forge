import atexit
import os
import unittest
import warnings
import numpy as np
import pytest
from scipy import sparse
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import _IS_WASM
from sklearn.utils._testing import (
from sklearn.utils.deprecation import deprecated
from sklearn.utils.fixes import (
from sklearn.utils.metaestimators import available_if
def f_too_many_param_docstring(a, b):
    """Function f

    Parameters
    ----------
    a : int
        Parameter a
    b : int
        Parameter b
    c : int
        Parameter c

    Returns
    -------
    d : list
        Parameter c
    """
    d = a + b
    return d