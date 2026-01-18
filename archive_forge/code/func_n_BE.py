import numpy as np
from ase.parallel import world
from ase import units
from ase.md.md import process_temperature
def n_BE(temp, omega):
    """Bose-Einstein distribution function.

    Args:
        temp: temperature converted to eV (*units.kB)
        omega: sequence of frequencies converted to eV

    Returns:
        Value of Bose-Einstein distribution function for each energy

    """
    omega = np.asarray(omega)
    if temp < eps_temp:
        n = np.zeros_like(omega)
    else:
        n = 1 / (np.exp(omega / temp) - 1)
    return n