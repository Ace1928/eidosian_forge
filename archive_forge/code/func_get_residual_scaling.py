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
def get_residual_scaling(self):
    eq_scaling = self._ex_model.get_equality_constraint_scaling_factors()
    output_con_scaling = self._ex_model.get_output_constraint_scaling_factors()
    if eq_scaling is None and output_con_scaling is None:
        return None
    if eq_scaling is None:
        eq_scaling = np.ones(self._ex_model.n_equality_constraints())
    if output_con_scaling is None:
        output_con_scaling = np.ones(self._ex_model.n_outputs())
    return np.concatenate((eq_scaling, output_con_scaling))