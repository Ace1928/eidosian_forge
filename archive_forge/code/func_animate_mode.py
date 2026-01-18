import inspect
import json
import numpy as np
from ase.data import covalent_radii
from ase.neighborlist import NeighborList
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.utilities import atoms_too_close, gather_atoms_by_tag
from scipy.spatial.distance import cdist
def animate_mode(self, atoms, mode, nim=30, amplitude=1.0):
    """Returns an Atoms object showing an animation of the mode."""
    pos = atoms.get_positions()
    mode = mode.reshape(np.shape(pos))
    animation = []
    for i in range(nim):
        newpos = pos + amplitude * mode * np.sin(i * 2 * np.pi / nim)
        image = atoms.copy()
        image.positions = newpos
        animation.append(image)
    return animation