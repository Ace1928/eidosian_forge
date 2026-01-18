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
def evaluate_residuals(self):
    resid_list = []
    if self._ex_model.n_equality_constraints() > 0:
        resid_list.append(self._ex_model.evaluate_equality_constraints())
    if self._ex_model.n_outputs() > 0:
        computed_output_values = self._ex_model.evaluate_outputs()
        output_resid = computed_output_values - self._output_values
        resid_list.append(output_resid)
    return np.concatenate(resid_list)