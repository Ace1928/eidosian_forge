from time import time
from math import sqrt, pi
import numpy as np
from ase.parallel import paropen
from ase.dft.kpoints import get_monkhorst_pack_size_and_offset
from ase.transport.tools import dagger, normalize
from ase.io.jsonio import read_json, write_json
def get_hamiltonian(self, k=0):
    """Get Hamiltonian at existing k-vector of index k

        ::

                  dag
          H(k) = V    diag(eps )  V
                  k           k    k
        """
    eps_n = self.calc.get_eigenvalues(kpt=k, spin=self.spin)[:self.nbands]
    return np.dot(dag(self.V_knw[k]) * eps_n, self.V_knw[k])