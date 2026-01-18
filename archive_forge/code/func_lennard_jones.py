import numpy as np
from ase.calculators.calculator import Calculator
from ase.calculators.qmmm import combine_lj_lorenz_berthelot
from ase import units
import copy
def lennard_jones(self, atoms1, atoms2):
    pos1 = atoms1.get_positions().reshape((-1, self.apm1, 3))
    pos2 = atoms2.get_positions().reshape((-1, self.apm2, 3))
    f1 = np.zeros_like(atoms1.positions)
    f2 = np.zeros_like(atoms2.positions)
    energy = 0.0
    cell = self.cell.diagonal()
    for q, p1 in enumerate(pos1):
        eps = self.epsilon
        sig = self.sigma
        R00 = pos2[:, 0] - p1[0, :]
        shift = np.zeros_like(R00)
        for i, periodic in enumerate(self.pbc):
            if periodic:
                L = cell[i]
                shift[:, i] = (R00[:, i] + L / 2) % L - L / 2 - R00[:, i]
        R00 += shift
        d002 = (R00 ** 2).sum(1)
        d00 = d002 ** 0.5
        x1 = d00 > self.rc - self.width
        x2 = d00 < self.rc
        x12 = np.logical_and(x1, x2)
        y = (d00[x12] - self.rc + self.width) / self.width
        t = np.zeros(len(d00))
        t[x2] = 1.0
        t[x12] -= y ** 2 * (3.0 - 2.0 * y)
        dt = np.zeros(len(d00))
        dt[x12] -= 6.0 / self.width * y * (1.0 - y)
        for qa in range(len(p1)):
            if ~np.any(eps[qa, :]):
                continue
            R = pos2 - p1[qa, :] + shift[:, None]
            d2 = (R ** 2).sum(2)
            c6 = (sig[qa, :] ** 2 / d2) ** 3
            c12 = c6 ** 2
            e = 4 * eps[qa, :] * (c12 - c6)
            energy += np.dot(e.sum(1), t)
            f = t[:, None, None] * (24 * eps[qa, :] * (2 * c12 - c6) / d2)[:, :, None] * R
            f00 = -(e.sum(1) * dt / d00)[:, None] * R00
            f2 += f.reshape((-1, 3))
            f1[q * self.apm1 + qa, :] -= f.sum(0).sum(0)
            f1[q * self.apm1, :] -= f00.sum(0)
            f2[::self.apm2, :] += f00
    return (energy, f1, f2)