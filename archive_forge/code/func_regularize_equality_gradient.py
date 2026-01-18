from abc import ABCMeta, abstractmethod
from pyomo.contrib.pynumero.interfaces import pyomo_nlp, ampl_nlp
from pyomo.contrib.pynumero.sparse import BlockMatrix, BlockVector
import numpy as np
import scipy.sparse
from pyomo.common.timing import HierarchicalTimer
def regularize_equality_gradient(self, kkt, coef, copy_kkt=True):
    if copy_kkt:
        kkt = kkt.copy()
    reg_coef = coef
    ptb = reg_coef * scipy.sparse.identity(self._nlp.n_eq_constraints(), format='coo')
    kkt.set_block(2, 2, ptb)
    return kkt