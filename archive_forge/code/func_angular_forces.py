import os
import numpy as np
from ase.neighborlist import NeighborList
from ase.calculators.calculator import Calculator, all_changes
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from ase.units import Bohr, Hartree
def angular_forces(self, mu_i, mu, lam_i, lam, r, rvec, form1, form2):
    psi = np.zeros(mu.shape)
    for gamma in range(3):
        term1 = (mu_i[gamma] - mu[:, gamma]) * self.d[form1][form2](r)
        term2 = np.sum((mu_i - mu) * self.d_d[form1][form2](r)[:, np.newaxis] * (rvec * rvec[:, gamma][:, np.newaxis] / r[:, np.newaxis]), axis=1)
        term3 = 2 * np.sum((lam_i[:, gamma] + lam[:, :, gamma]) * rvec * self.q[form1][form2](r)[:, np.newaxis], axis=1)
        term4 = 0.0
        for alpha in range(3):
            for beta in range(3):
                rs = rvec[:, alpha] * rvec[:, beta] * rvec[:, gamma]
                term4 += (lam_i[alpha, beta] + lam[:, alpha, beta]) * self.d_q[form1][form2](r) * rs / r
        term5 = (lam_i.trace() + lam.trace(axis1=1, axis2=2)) * (self.d_q[form1][form2](r) * r + 2 * self.q[form1][form2](r)) * rvec[:, gamma] / 3.0
        psi[:, gamma] = term1 + term2 + term3 + term4 - term5
    return np.sum(psi, axis=0)