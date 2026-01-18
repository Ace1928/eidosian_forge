import itertools
import collections
import numpy as np
from ase import Atoms
from ase.geometry.cell import complete_cell
from ase.geometry.dimensionality import analyze_dimensionality
from ase.geometry.dimensionality import rank_determination
from ase.geometry.dimensionality.bond_generator import next_bond
from ase.geometry.dimensionality.interval_analysis import merge_intervals
def construct_inplane_basis(atoms, k, v):
    basis_points = np.array([offset for c, offset in v if c == k])
    assert len(basis_points) >= 3
    assert (0, 0, 0) in [tuple(e) for e in basis_points]
    sizes = np.linalg.norm(basis_points, axis=1)
    indices = np.argsort(sizes)
    basis_points = basis_points[indices]
    best = (float('inf'), None)
    for u, v in itertools.combinations(basis_points, 2):
        basis = np.array([[0, 0, 0], u, v])
        if np.linalg.matrix_rank(basis) < 2:
            continue
        a = np.dot(u, atoms.get_cell())
        b = np.dot(v, atoms.get_cell())
        norm = np.linalg.norm(np.cross(a, b))
        best = min(best, (norm, a, b), key=lambda x: x[0])
    _, a, b = best
    return (a, b, orthogonal_basis(a, b))