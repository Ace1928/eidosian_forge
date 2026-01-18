import warnings
import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.optimize._optimize import _status_message, _wrap_callback
from scipy._lib._util import check_random_state, MapWrapper, _FunctionWrapper
from scipy.optimize._constraints import (Bounds, new_bounds_to_old,
from scipy.sparse import issparse
def init_population_array(self, init):
    """
        Initializes the population with a user specified population.

        Parameters
        ----------
        init : np.ndarray
            Array specifying subset of the initial population. The array should
            have shape (S, N), where N is the number of parameters.
            The population is clipped to the lower and upper bounds.
        """
    popn = np.asarray(init, dtype=np.float64)
    if np.size(popn, 0) < 5 or popn.shape[1] != self.parameter_count or len(popn.shape) != 2:
        raise ValueError('The population supplied needs to have shape (S, len(x)), where S > 4.')
    self.population = np.clip(self._unscale_parameters(popn), 0, 1)
    self.num_population_members = np.size(self.population, 0)
    self.population_shape = (self.num_population_members, self.parameter_count)
    self.population_energies = np.full(self.num_population_members, np.inf)
    self._nfev = 0