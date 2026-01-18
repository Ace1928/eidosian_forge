import numpy as np
from ase.parallel import world, broadcast
from ase.geometry import get_distances
def attach_randomly(atoms1, atoms2, distance, rng=np.random):
    """Randomly attach two structures with a given minimal distance

    Parameters
    ----------
    atoms1: Atoms object
    atoms2: Atoms object
    distance: float
      Required distance
    rng: random number generator object
      defaults to np.random.RandomState()

    Returns
    -------
    Joined structure as an atoms object.
    """
    atoms2 = atoms2.copy()
    atoms2.rotate('x', random_unit_vector(rng), center=atoms2.get_center_of_mass())
    return attach(atoms1, atoms2, distance, direction=random_unit_vector(rng))