from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
def get_zpq_triple(self, key_z):
    """
        Gives a flattening as triple [z;p,q] representing an element
        in the generalized Extended Bloch group similar to the way the
        triple [z;p,q] is used in Lemma 3.2 in
        Walter D. Neumann, Extended Bloch group and the Cheeger-Chern-Simons class
        http://arxiv.org/abs/math.GT/0307092
        """
    if not key_z[:2] == 'z_':
        raise Exception('Need to be called with cross ratio variable z_....')
    key_zp = 'zp_' + key_z[2:]
    w, z, p = self[key_z]
    wp, zp, q_canonical_branch_cut = self[key_zp]
    pari_z = _convert_to_pari_float(z)
    f = pari('2 * Pi * I') / self._evenN
    q_dilog_branch_cut = ((wp + (1 - pari_z).log()) / f).round()
    return (z, p, q_dilog_branch_cut)