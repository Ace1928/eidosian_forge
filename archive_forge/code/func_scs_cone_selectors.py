import numpy as np
from scipy import sparse
from cvxpy.atoms.suppfunc import SuppFuncAtom
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.cvx_attr2constr import CONVEX_ATTRIBUTES
def scs_cone_selectors(K):
    """
    Parse a ConeDims object, as returned from SCS's apply function.

    Return a dictionary which gives row-wise information for the affine
    operator returned from SCS's apply function.

    Parameters
    ----------
    K : cvxpy.reductions.solvers.conic_solver.ConeDims

    Returns
    -------
    selectors : dict
        Keyed by strings, which specify cone types. Values are numpy
        arrays, or lists of numpy arrays. The numpy arrays give row indices
        of the affine operator (A, b) returned by SCS's apply function.
    """
    if K.p3d:
        msg = "SuppFunc doesn't yet support feasible sets represented \n"
        msg += 'with power cone constraints.'
        raise NotImplementedError(msg)
    idx = K.zero
    nonneg_idxs = np.arange(idx, idx + K.nonneg)
    idx += K.nonneg
    soc_idxs = []
    for soc in K.soc:
        idxs = np.arange(idx, idx + soc)
        soc_idxs.append(idxs)
        idx += soc
    psd_idxs = []
    for psd in K.psd:
        veclen = psd * (psd + 1) // 2
        psd_idxs.append(np.arange(idx, idx + veclen))
        idx += veclen
    expsize = 3 * K.exp
    exp_idxs = np.arange(idx, idx + expsize)
    selectors = {'nonneg': nonneg_idxs, 'exp': exp_idxs, 'soc': soc_idxs, 'psd': psd_idxs}
    return selectors