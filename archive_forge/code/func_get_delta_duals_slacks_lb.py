from abc import ABCMeta, abstractmethod
from pyomo.contrib.pynumero.interfaces import pyomo_nlp, ampl_nlp
from pyomo.contrib.pynumero.sparse import BlockMatrix, BlockVector
import numpy as np
import scipy.sparse
from pyomo.common.timing import HierarchicalTimer
def get_delta_duals_slacks_lb(self):
    res = (self._barrier - self._duals_slacks_lb * self._delta_slacks) / (self._slacks - self._nlp.ineq_lb()) - self._duals_slacks_lb
    return res