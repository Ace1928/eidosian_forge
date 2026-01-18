from warnings import warn
import numpy as np
from numpy import asarray
from scipy.sparse import (issparse,
from scipy.sparse._sputils import is_pydata_spmatrix
from scipy.linalg import LinAlgError
import copy
from . import _superlu
def csc_construct_func(*a, cls=type(A)):
    return cls(csc_matrix(*a))