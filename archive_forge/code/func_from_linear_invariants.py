from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from itertools import repeat, chain
import warnings
import numpy as np
from .util import import_
from .core import ODESys, RecoverableError
from .util import (
@classmethod
def from_linear_invariants(cls, ori_sys, preferred=None, **kwargs):
    """ Reformulates the ODE system in fewer variables.

        Given linear invariant equations one can always reduce the number
        of dependent variables in the system by the rank of the matrix describing
        this linear system.

        Parameters
        ----------
        ori_sys : :class:`SymbolicSys` instance
        preferred : iterable of preferred dependent variables
            Due to numerical rounding it is preferable to choose the variables
            which are expected to be of the largest magnitude during integration.
        \\*\\*kwargs :
            Keyword arguments passed on to constructor.
        """
    _be = ori_sys.be
    A = _be.Matrix(ori_sys.linear_invariants)
    rA, pivots = A.rref()
    if len(pivots) < A.shape[0]:
        raise NotImplementedError('Linear invariants contain linear dependencies.')
    per_row_cols = [(ri, [ci for ci in range(A.cols) if A[ri, ci] != 0]) for ri in range(A.rows)]
    if preferred is None:
        preferred = ori_sys.names[:A.rows] if ori_sys.dep_by_name else list(range(A.rows))
    targets = [ori_sys.names.index(dep) if ori_sys.dep_by_name else dep if isinstance(dep, int) else ori_sys.dep.index(dep) for dep in preferred]
    row_tgt = []
    for ri, colids in sorted(per_row_cols, key=lambda k: len(k[1])):
        for tgt in targets:
            if tgt in colids:
                row_tgt.append((ri, tgt))
                targets.remove(tgt)
                break
        if len(targets) == 0:
            break
    else:
        raise ValueError('Could not find a solutions for: %s' % targets)

    def analytic_factory(x0, y0, p0, be):
        return {ori_sys.dep[tgt]: y0[ori_sys.dep[tgt] if ori_sys.dep_by_name else tgt] - sum([A[ri, ci] * (ori_sys.dep[ci] - y0[ori_sys.dep[ci] if ori_sys.dep_by_name else ci]) for ci in range(A.cols) if ci != tgt]) / A[ri, tgt] for ri, tgt in row_tgt}
    ori_li_nms = ori_sys.linear_invariant_names or ()
    new_lin_invar = [[cell for ci, cell in enumerate(row) if ci not in list(zip(*row_tgt))[1]] for ri, row in enumerate(A.tolist()) if ri not in list(zip(*row_tgt))[0]]
    new_lin_i_nms = [nam for ri, nam in enumerate(ori_li_nms) if ri not in list(zip(*row_tgt))[0]]
    return cls(ori_sys, analytic_factory, linear_invariants=new_lin_invar, linear_invariant_names=new_lin_i_nms, **kwargs)