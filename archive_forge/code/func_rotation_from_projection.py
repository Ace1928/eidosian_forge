from time import time
from math import sqrt, pi
import numpy as np
from ase.parallel import paropen
from ase.dft.kpoints import get_monkhorst_pack_size_and_offset
from ase.transport.tools import dagger, normalize
from ase.io.jsonio import read_json, write_json
def rotation_from_projection(proj_nw, fixed, ortho=True):
    """Determine rotation and coefficient matrices from projections

    proj_nw = <psi_n|p_w>
    psi_n: eigenstates
    p_w: localized function

    Nb (n) = Number of bands
    Nw (w) = Number of wannier functions
    M  (f) = Number of fixed states
    L  (l) = Number of extra degrees of freedom
    U  (u) = Number of non-fixed states
    """
    Nb, Nw = proj_nw.shape
    M = fixed
    L = Nw - M
    U_ww = np.empty((Nw, Nw), dtype=proj_nw.dtype)
    U_ww[:M] = proj_nw[:M]
    if L > 0:
        proj_uw = proj_nw[M:]
        eig_w, C_ww = np.linalg.eigh(np.dot(dag(proj_uw), proj_uw))
        C_ul = np.dot(proj_uw, C_ww[:, np.argsort(-eig_w.real)[:L]])
        U_ww[M:] = np.dot(dag(C_ul), proj_uw)
    else:
        C_ul = np.empty((Nb - M, 0))
    normalize(C_ul)
    if ortho:
        lowdin(U_ww)
    else:
        normalize(U_ww)
    return (U_ww, C_ul)