import sys
import numpy as np
from itertools import combinations_with_replacement
import ase.units as u
from ase.parallel import parprint, paropen
from ase.vibrations.resonant_raman import ResonantRaman
from ase.vibrations.franck_condon import FranckCondonOverlap
from ase.vibrations.franck_condon import FranckCondonRecursive
def meA(self, omega, gamma=0.1):
    """Evaluate Albrecht A term.

        Returns
        -------
        Full Albrecht A matrix element. Unit: e^2 Angstrom^2 / eV
        """
    self.read()
    if not hasattr(self, 'fcr'):
        self.fcr = FranckCondonRecursive()
    omL = omega + 1j * gamma
    omS_Q = omL - self.om_Q
    n_p, myp, exF_pr = self.init_parallel_excitations()
    exF_pr = np.where(np.abs(exF_pr) > 0.01, exF_pr, 0)
    m_Qcc = np.zeros((self.ndof, 3, 3), dtype=complex)
    for p in myp:
        energy = self.ex0E_p[p]
        d_Q = self.unitless_displacements(exF_pr[p])
        energy_Q = energy - self.om_Q * d_Q ** 2 / 2.0
        me_cc = np.outer(self.ex0m_pc[p], self.ex0m_pc[p].conj())
        wm_Q = np.zeros(self.ndof, dtype=complex)
        wp_Q = np.zeros(self.ndof, dtype=complex)
        for m in range(self.nm):
            fco_Q = self.fcr.direct0mm1(m, d_Q)
            e_Q = energy_Q + m * self.om_Q
            wm_Q += fco_Q / (e_Q - omL)
            wp_Q += fco_Q / (e_Q + omS_Q)
        m_Qcc += np.einsum('a,bc->abc', wm_Q, me_cc)
        m_Qcc += np.einsum('a,bc->abc', wp_Q, me_cc.conj())
    self.comm.sum(m_Qcc)
    return m_Qcc