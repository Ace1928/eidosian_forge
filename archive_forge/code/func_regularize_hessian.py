from abc import ABCMeta, abstractmethod
from pyomo.contrib.pynumero.interfaces import pyomo_nlp, ampl_nlp
from pyomo.contrib.pynumero.sparse import BlockMatrix, BlockVector
import numpy as np
import scipy.sparse
from pyomo.common.timing import HierarchicalTimer
def regularize_hessian(self, kkt, coef, copy_kkt=True):
    if copy_kkt:
        kkt = kkt.copy()
    hess = kkt.get_block(0, 0)
    ptb = coef * scipy.sparse.identity(self._nlp.n_primals(), format='coo')
    hess += ptb
    kkt.set_block(0, 0, hess)
    return kkt