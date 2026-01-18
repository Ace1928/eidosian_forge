import numpy as np
from ase.units import Bohr, Hartree
from ase.calculators.calculator import Calculator
from scipy.special import erfinv, erfc
from ase.neighborlist import neighbor_list
from ase.parallel import world
from ase.utils import IOContext
def damping(self, RAB, R0A, R0B, d=20, sR=0.94):
    """Damping factor.

        Standard values for d and sR as given in
        Tkatchenko and Scheffler PRL 102 (2009) 073005."""
    scale = 1.0 / (sR * (R0A + R0B))
    x = RAB * scale
    chi = np.exp(-d * (x - 1.0))
    return (1.0 / (1.0 + chi), d * scale * chi / (1.0 + chi) ** 2)