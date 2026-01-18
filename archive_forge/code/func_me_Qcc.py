import sys
import numpy as np
from itertools import combinations_with_replacement
import ase.units as u
from ase.parallel import parprint, paropen
from ase.vibrations.resonant_raman import ResonantRaman
from ase.vibrations.franck_condon import FranckCondonOverlap
from ase.vibrations.franck_condon import FranckCondonRecursive
def me_Qcc(self, omega, gamma):
    """Full matrix element"""
    self.read()
    approx = self.approximation.lower()
    nv = len(self.om_v)
    V_vcc = np.zeros((nv, 3, 3), dtype=complex)
    if approx == 'albrecht a' or approx == 'albrecht':
        if self.combinations == 1:
            V_vcc += self.meA(omega, gamma)[self.skip:]
        else:
            V_vcc += self.meAmult(omega, gamma)
    if approx == 'albrecht bc' or approx == 'albrecht':
        if self.combinations == 1:
            vel_vcc = self.meBC(omega, gamma)
            V_vcc += vel_vcc * self.vib01_Q[:, None, None]
        else:
            vel_vcc = self.meBCmult(omega, gamma)
            V_vcc = 0
    elif approx == 'albrecht b':
        assert self.combinations == 1
        vel_vcc = self.meBC(omega, gamma, term='B')
        V_vcc = vel_vcc * self.vib01_Q[:, None, None]
    if approx == 'albrecht c':
        assert self.combinations == 1
        vel_vcc = self.meBC(omega, gamma, term='C')
        V_vcc = vel_vcc * self.vib01_Q[:, None, None]
    return V_vcc