from math import sqrt, exp, log
import numpy as np
from ase.data import chemical_symbols, atomic_numbers
from ase.units import Bohr
from ase.neighborlist import NeighborList
from ase.calculators.calculator import (Calculator, all_changes,
def interact1(self, a1, a2, d, r, p1, p2, ksi):
    x = exp(self.acut * (r - self.rc))
    theta = 1.0 / (1.0 + x)
    y1 = 0.5 * p1['V0'] * exp(-p2['kappa'] * (r / beta - p2['s0'])) * ksi / p1['gamma2'] * theta
    y2 = 0.5 * p2['V0'] * exp(-p1['kappa'] * (r / beta - p1['s0'])) / ksi / p2['gamma2'] * theta
    self.energy -= y1 + y2
    self.energies[a1] -= (y1 + y2) / 2
    self.energies[a2] -= (y1 + y2) / 2
    f = ((y1 * p2['kappa'] + y2 * p1['kappa']) / beta + (y1 + y2) * self.acut * theta * x) * d / r
    self.forces[a1] += f
    self.forces[a2] -= f
    self.stress -= np.outer(f, d)
    self.sigma1[a1] += exp(-p2['eta2'] * (r - beta * p2['s0'])) * ksi * theta / p1['gamma1']
    self.sigma1[a2] += exp(-p1['eta2'] * (r - beta * p1['s0'])) / ksi * theta / p2['gamma1']