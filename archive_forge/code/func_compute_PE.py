import warnings
import numpy as np
import pandas as pd
from scipy import stats
def compute_PE(A, B):
    """ Computes the probability of excedance for each row of the
        Cohn dataframe. """
    N = len(A)
    PE = np.empty(N, dtype='float64')
    PE[-1] = 0.0
    for j in range(N - 2, -1, -1):
        PE[j] = PE[j + 1] + (1 - PE[j + 1]) * A[j] / (A[j] + B[j])
    return PE