import numpy as np
import ase.units as u
from ase.vibrations.raman import Raman, RamanPhonons
from ase.vibrations.resonant_raman import ResonantRaman
from ase.calculators.excitation_list import polarizability
def read_excitations(self):
    """Read excitations from files written"""
    self.al0_rr = None
    self.alm_rr = []
    self.alp_rr = []
    for a, i in zip(self.myindices, self.myxyz):
        for sign, al_rr in zip([-1, 1], [self.alm_rr, self.alp_rr]):
            disp = self._disp(a, i, sign)
            al_rr.append(disp.load_static_polarizability())