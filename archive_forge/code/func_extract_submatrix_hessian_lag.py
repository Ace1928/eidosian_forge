import os
import numpy as np
from scipy.sparse import coo_matrix
from pyomo.common.deprecation import deprecated
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt import WriterFactory
import pyomo.core.base as pyo
from pyomo.common.collections import ComponentMap
from pyomo.common.env import CtypesEnviron
from ..sparse.block_matrix import BlockMatrix
from pyomo.contrib.pynumero.interfaces.ampl_nlp import AslNLP
from pyomo.contrib.pynumero.interfaces.nlp import NLP
from .external_grey_box import ExternalGreyBoxBlock
def extract_submatrix_hessian_lag(self, pyomo_variables_rows, pyomo_variables_cols):
    """
        Return the submatrix of the hessian of the lagrangian that
        corresponds to the list of Pyomo variables provided

        Parameters
        ----------
        pyomo_variables_rows : list of Pyomo Var or VarData objects
            List of Pyomo Var or VarData objects corresponding to the desired rows
        pyomo_variables_cols : list of Pyomo Var or VarData objects
            List of Pyomo Var or VarData objects corresponding to the desired columns
        """
    hess_lag = self.evaluate_hessian_lag()
    primal_indices_rows = self.get_primal_indices(pyomo_variables_rows)
    primal_indices_cols = self.get_primal_indices(pyomo_variables_cols)
    row_mask = np.isin(hess_lag.row, primal_indices_rows)
    col_mask = np.isin(hess_lag.col, primal_indices_cols)
    submatrix_mask = row_mask & col_mask
    submatrix_irows = np.compress(submatrix_mask, hess_lag.row)
    submatrix_jcols = np.compress(submatrix_mask, hess_lag.col)
    submatrix_data = np.compress(submatrix_mask, hess_lag.data)
    submatrix_map = {j: i for i, j in enumerate(primal_indices_rows)}
    for i, v in enumerate(submatrix_irows):
        submatrix_irows[i] = submatrix_map[v]
    submatrix_map = {j: i for i, j in enumerate(primal_indices_cols)}
    for i, v in enumerate(submatrix_jcols):
        submatrix_jcols[i] = submatrix_map[v]
    return coo_matrix((submatrix_data, (submatrix_irows, submatrix_jcols)), shape=(len(primal_indices_rows), len(primal_indices_cols)))