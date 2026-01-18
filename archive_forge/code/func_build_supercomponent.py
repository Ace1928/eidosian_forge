import itertools
import collections
import numpy as np
from ase import Atoms
from ase.geometry.cell import complete_cell
from ase.geometry.dimensionality import analyze_dimensionality
from ase.geometry.dimensionality import rank_determination
from ase.geometry.dimensionality.bond_generator import next_bond
from ase.geometry.dimensionality.interval_analysis import merge_intervals
def build_supercomponent(atoms, components, k, v, anchor=True):
    positions = []
    numbers = []
    for c, offset in dict(v[::-1]).items():
        indices = np.where(components == c)[0]
        ps = atoms.positions + np.dot(offset, atoms.get_cell())
        positions.extend(ps[indices])
        numbers.extend(atoms.numbers[indices])
    positions = np.array(positions)
    numbers = np.array(numbers)
    anchor_index = next((i for i in range(len(atoms)) if components[i] == k))
    if anchor:
        positions -= atoms.positions[anchor_index]
    return (positions, numbers)