import numpy as np
from scipy import stats
from scipy.special import factorial
from statsmodels.base.model import GenericLikelihoodModel
def maxabsrel(arr1, arr2):
    return np.max(np.abs(arr2 / arr1 - 1))