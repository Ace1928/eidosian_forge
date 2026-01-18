import numpy as np
import itertools
from scipy import sparse as sp
from scipy.spatial import cKDTree
import scipy.sparse.csgraph as csgraph
from ase.data import atomic_numbers, covalent_radii
from ase.geometry import complete_cell, find_mic, wrap_positions
from ase.geometry import minkowski_reduce
from ase.cell import Cell
def natural_cutoffs(atoms, mult=1, **kwargs):
    """Generate a radial cutoff for every atom based on covalent radii.

    The covalent radii are a reasonable cutoff estimation for bonds in
    many applications such as neighborlists, so function generates an
    atoms length list of radii based on this idea.

    * atoms: An atoms object
    * mult: A multiplier for all cutoffs, useful for coarse grained adjustment
    * kwargs: Symbol of the atom and its corresponding cutoff, used to override the covalent radii
    """
    return [kwargs.get(atom.symbol, covalent_radii[atom.number] * mult) for atom in atoms]