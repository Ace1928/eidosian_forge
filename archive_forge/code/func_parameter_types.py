import numpy as np
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.gaussian_process.kernels import Kernel
import inspect
@classmethod
def parameter_types(cls):
    """Return the argument and return types of an operator.

            Parameters
            ----------
            None

            Returns
            -------
            parameter_types: tuple
                Tuple of the DEAP parameter types and the DEAP return type for the
                operator

            """
    return ([np.ndarray] + arg_types, np.ndarray)