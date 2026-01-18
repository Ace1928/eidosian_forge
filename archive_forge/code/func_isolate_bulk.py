import itertools
import collections
import numpy as np
from ase import Atoms
from ase.geometry.cell import complete_cell
from ase.geometry.dimensionality import analyze_dimensionality
from ase.geometry.dimensionality import rank_determination
from ase.geometry.dimensionality.bond_generator import next_bond
from ase.geometry.dimensionality.interval_analysis import merge_intervals
def isolate_bulk(atoms, components, k, v):
    positions, numbers = build_supercomponent(atoms, components, k, v, anchor=False)
    atoms = Atoms(numbers=numbers, positions=positions, cell=atoms.cell, pbc=[1, 1, 1])
    atoms.wrap(eps=0)
    return atoms