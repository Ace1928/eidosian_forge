from scipy.sparse import coo_matrix
import os
import numpy as np
from pyomo.common.deprecation import deprecated
from pyomo.contrib.pynumero.interfaces.nlp import ExtendedNLP
def create_new_vector(self, vector_type):
    """
        Creates a vector of the appropriate length and structure as
        requested

        Parameters
        ----------
        vector_type: {'primals', 'constraints', 'eq_constraints', 'ineq_constraints',
                      'duals', 'duals_eq', 'duals_ineq'}
            String identifying the appropriate  vector  to create.

        Returns
        -------
        numpy.ndarray
        """
    if vector_type == 'primals':
        return np.zeros(self.n_primals(), dtype=np.float64)
    elif vector_type == 'constraints' or vector_type == 'duals':
        return np.zeros(self.n_constraints(), dtype=np.float64)
    elif vector_type == 'eq_constraints' or vector_type == 'duals_eq':
        return np.zeros(self.n_eq_constraints(), dtype=np.float64)
    elif vector_type == 'ineq_constraints' or vector_type == 'duals_ineq':
        return np.zeros(self.n_ineq_constraints(), dtype=np.float64)
    else:
        raise RuntimeError('Called create_new_vector with an unknown vector_type')