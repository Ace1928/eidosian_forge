from time import time
from math import sqrt, pi
import numpy as np
from ase.parallel import paropen
from ase.dft.kpoints import get_monkhorst_pack_size_and_offset
from ase.transport.tools import dagger, normalize
from ase.io.jsonio import read_json, write_json
def get_hopping(self, R):
    """Returns the matrix H(R)_nm=<0,n|H|R,m>.

        ::

                                1   _   -ik.R
          H(R) = <0,n|H|R,m> = --- >_  e      H(k)
                                Nk  k

        where R is the cell-distance (in units of the basis vectors of
        the small cell) and n,m are indices of the Wannier functions.
        """
    H_ww = np.zeros([self.nwannier, self.nwannier], complex)
    for k, kpt_c in enumerate(self.kpt_kc):
        phase = np.exp(-2j * pi * np.dot(np.array(R), kpt_c))
        H_ww += self.get_hamiltonian(k) * phase
    return H_ww / self.Nk