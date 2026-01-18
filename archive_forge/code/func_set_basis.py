import numpy as np
from ase.data import reference_states as _refstate
from ase.cluster.factory import ClusterFactory
def set_basis(self):
    a = self.lattice_constant
    if not isinstance(a, (int, float)):
        raise ValueError('Improper lattice constant for %s crystal.' % (self.xtal_name,))
    self.lattice_basis = np.array([[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]])
    self.resiproc_basis = self.get_resiproc_basis(self.lattice_basis)