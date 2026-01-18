import numpy as np
from ase.units import Hartree, Bohr
def get_dipole_tensor(self, form='r'):
    """Return the oscillator strength tensor"""
    me = self.get_dipole_me(form)
    return 2 * np.outer(me, me.conj()) * self.energy / Hartree