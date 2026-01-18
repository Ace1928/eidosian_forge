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
def extract_subvector_constraints(self, pyomo_constraints):
    """
        Return the values of the constraints
        corresponding to the list of Pyomo constraints provided

        Parameters
        ----------
        pyomo_constraints : list of Pyomo Constraint or ConstraintData objects
        """
    residuals = self.evaluate_constraints()
    return residuals[self.get_constraint_indices(pyomo_constraints)]