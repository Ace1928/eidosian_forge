import numpy as np
import ase.units as u
from ase.vibrations.raman import Raman, RamanPhonons
from ase.vibrations.resonant_raman import ResonantRaman
from ase.calculators.excitation_list import polarizability
def kappa_cc(me_pc, e_p, omega, gamma, form='v'):
    """Kappa tensor after Profeta and Mauri
            PRB 63 (2001) 245415"""
    k_cc = np.zeros((3, 3), dtype=complex)
    for p, me_c in enumerate(me_pc):
        me_cc = np.outer(me_c, me_c.conj())
        k_cc += me_cc / (e_p[p] - omega - 1j * gamma)
        if self.nonresonant:
            k_cc += me_cc.conj() / (e_p[p] + omega + 1j * gamma)
    return k_cc