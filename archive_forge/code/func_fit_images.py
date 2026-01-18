import numpy as np
from collections import namedtuple
from ase.geometry import find_mic
def fit_images(images):
    """Fits a series of images with a smoothed line for producing a standard
    NEB plot. Returns a `ForceFit` data structure; the plot can be produced
    by calling the `plot` method of `ForceFit`."""
    R = [atoms.positions for atoms in images]
    E = [atoms.get_potential_energy() for atoms in images]
    F = [atoms.get_forces() for atoms in images]
    A = images[0].cell
    pbc = images[0].pbc
    return fit_raw(E, F, R, A, pbc)