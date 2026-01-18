from math import sqrt
from warnings import warn
import numpy as np
from scipy.linalg import expm, logm
from ase.calculators.calculator import PropertyNotImplementedError
from ase.geometry import (find_mic, wrap_positions, get_distances_derivatives,
from ase.utils.parsemath import eval_expression
from ase.stress import (full_3x3_to_voigt_6_stress,
def redistribute_forces_md(self, atoms, forces, rand=False):
    """Redistribute forces within a triple when performing molecular
        dynamics.

        When rand=True, use the equations for random force terms, as
        used e.g. by Langevin dynamics, otherwise apply the standard
        equations for deterministic forces (see Ciccotti et al. Molecular
        Physics 47 (1982))."""
    if self.bondlengths is None:
        self.initialize(atoms)
    forces_n, forces_m, forces_o = self.get_slices(forces)
    C1_1 = self.C1[:, 0, None]
    C1_2 = self.C1[:, 1, None]
    C2_1 = self.C2[:, 0, None]
    C2_2 = self.C2[:, 1, None]
    mass_nn = self.mass_n[:, None]
    mass_mm = self.mass_m[:, None]
    mass_oo = self.mass_o[:, None]
    if rand:
        mr1 = (mass_mm / mass_nn) ** 0.5
        mr2 = (mass_oo / mass_nn) ** 0.5
        mr3 = (mass_nn / mass_mm) ** 0.5
        mr4 = (mass_oo / mass_mm) ** 0.5
    else:
        mr1 = 1.0
        mr2 = 1.0
        mr3 = 1.0
        mr4 = 1.0
    fr_n = (1 - C1_1 * C2_1 * mass_oo * mass_mm) * forces_n - C2_1 * (C1_2 * mr1 * mass_oo * mass_nn * forces_m - mr2 * mass_mm * mass_nn * forces_o)
    fr_m = (1 - C1_2 * C2_2 * mass_oo * mass_nn) * forces_m - C2_2 * (C1_1 * mr3 * mass_oo * mass_mm * forces_n - mr4 * mass_mm * mass_nn * forces_o)
    self.set_slices(fr_n, fr_m, 0.0, forces)