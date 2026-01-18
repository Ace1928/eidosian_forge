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
def extract_subvector_grad_objective(self, pyomo_variables):
    """Compute the gradient of the objective and return the entries
        corresponding to the given Pyomo variables

        Parameters
        ----------
        pyomo_variables : list of Pyomo Var or VarData objects
        """
    grad_obj = self.evaluate_grad_objective()
    return grad_obj[self.get_primal_indices(pyomo_variables)]