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
def get_constraint_indices(self, pyomo_constraints):
    """
        Return the list of indices for the constraints
        corresponding to the list of Pyomo constraints provided

        Parameters
        ----------
        pyomo_constraints : list of Pyomo Constraint or ConstraintData objects
        """
    assert isinstance(pyomo_constraints, list)
    con_indices = []
    for c in pyomo_constraints:
        if c.is_indexed():
            for cd in c.values():
                con_id = self._condata_to_idx[cd]
                con_indices.append(con_id)
        else:
            con_id = self._condata_to_idx[c]
            con_indices.append(con_id)
    return con_indices