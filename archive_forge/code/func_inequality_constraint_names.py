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
def inequality_constraint_names(self):
    """
        Return an ordered list of the Pyomo ConData names in
        the order corresponding to the inequality constraints.
        """
    inequality_constraints = self.get_pyomo_inequality_constraints()
    return [v.getname(fully_qualified=True) for v in inequality_constraints]