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
def get_primal_indices(self, pyomo_variables):
    """
        Return the list of indices for the primals
        corresponding to the list of Pyomo variables provided

        Parameters
        ----------
        pyomo_variables : list of Pyomo Var or VarData objects
        """
    assert isinstance(pyomo_variables, list)
    var_indices = []
    for v in pyomo_variables:
        if v.is_indexed():
            for vd in v.values():
                var_id = self._vardata_to_idx[vd]
                var_indices.append(var_id)
        else:
            var_id = self._vardata_to_idx[v]
            var_indices.append(var_id)
    return var_indices