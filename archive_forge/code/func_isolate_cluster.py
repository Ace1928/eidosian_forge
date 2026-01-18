import itertools
import collections
import numpy as np
from ase import Atoms
from ase.geometry.cell import complete_cell
from ase.geometry.dimensionality import analyze_dimensionality
from ase.geometry.dimensionality import rank_determination
from ase.geometry.dimensionality.bond_generator import next_bond
from ase.geometry.dimensionality.interval_analysis import merge_intervals
def isolate_cluster(atoms, components, k, v):
    positions, numbers = build_supercomponent(atoms, components, k, v)
    positions -= np.min(positions, axis=0)
    cell = np.diag(np.max(positions, axis=0))
    return Atoms(numbers=numbers, positions=positions, cell=cell, pbc=False)