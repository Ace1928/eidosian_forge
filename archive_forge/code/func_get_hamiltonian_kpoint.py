from time import time
from math import sqrt, pi
import numpy as np
from ase.parallel import paropen
from ase.dft.kpoints import get_monkhorst_pack_size_and_offset
from ase.transport.tools import dagger, normalize
from ase.io.jsonio import read_json, write_json
def get_hamiltonian_kpoint(self, kpt_c):
    """Get Hamiltonian at some new arbitrary k-vector

        ::

                  _   ik.R
          H(k) = >_  e     H(R)
                  R

        Warning: This method moves all Wannier functions to cell (0, 0, 0)
        """
    if self.verbose:
        print('Translating all Wannier functions to cell (0, 0, 0)')
    self.translate_all_to_cell()
    max = (self.kptgrid - 1) // 2
    N1, N2, N3 = max
    Hk = np.zeros([self.nwannier, self.nwannier], complex)
    for n1 in range(-N1, N1 + 1):
        for n2 in range(-N2, N2 + 1):
            for n3 in range(-N3, N3 + 1):
                R = np.array([n1, n2, n3], float)
                hop_ww = self.get_hopping(R)
                phase = np.exp(+2j * pi * np.dot(R, kpt_c))
                Hk += hop_ww * phase
    return Hk