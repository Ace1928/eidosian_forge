from abc import ABCMeta, abstractmethod
from pyomo.contrib.pynumero.interfaces import pyomo_nlp, ampl_nlp
from pyomo.contrib.pynumero.sparse import BlockMatrix, BlockVector
import numpy as np
import scipy.sparse
from pyomo.common.timing import HierarchicalTimer
def init_duals_primals_ub(self):
    return self._init_duals_primals_ub