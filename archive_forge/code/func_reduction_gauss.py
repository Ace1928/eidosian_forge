import itertools
import numpy as np
from ase.utils import pbc2pbc
from ase.cell import Cell
def reduction_gauss(B, hu, hv):
    """Calculate a Gauss-reduced lattice basis (2D reduction)."""
    cycle_checker = CycleChecker(d=2)
    u = hu @ B
    v = hv @ B
    for it in range(MAX_IT):
        x = int(round(np.dot(u, v) / np.dot(u, u)))
        hu, hv = (hv - x * hu, hu)
        u = hu @ B
        v = hv @ B
        site = np.array([hu, hv])
        if np.dot(u, u) >= np.dot(v, v) or cycle_checker.add_site(site):
            return (hv, hu)
    raise RuntimeError(f'Gaussian basis not found after {MAX_IT} iterations')