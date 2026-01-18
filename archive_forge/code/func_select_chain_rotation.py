import itertools
import collections
import numpy as np
from ase import Atoms
from ase.geometry.cell import complete_cell
from ase.geometry.dimensionality import analyze_dimensionality
from ase.geometry.dimensionality import rank_determination
from ase.geometry.dimensionality.bond_generator import next_bond
from ase.geometry.dimensionality.interval_analysis import merge_intervals
def select_chain_rotation(scaled):
    best = (-1, [1, 0, 0])
    for s in scaled:
        vhat = np.array([s[0], s[1], 0])
        norm = np.linalg.norm(vhat)
        if norm < 1e-06:
            continue
        vhat /= norm
        obj = np.sum(np.dot(scaled, vhat) ** 2)
        best = max(best, (obj, vhat), key=lambda x: x[0])
    _, vhat = best
    cost, sint, _ = vhat
    rot = np.array([[cost, -sint, 0], [sint, cost, 0], [0, 0, 1]])
    return np.dot(scaled, rot)