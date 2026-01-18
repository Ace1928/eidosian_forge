import itertools
import collections
import numpy as np
from ase import Atoms
from ase.geometry.cell import complete_cell
from ase.geometry.dimensionality import analyze_dimensionality
from ase.geometry.dimensionality import rank_determination
from ase.geometry.dimensionality.bond_generator import next_bond
from ase.geometry.dimensionality.interval_analysis import merge_intervals
def isolate_chain(atoms, components, k, v):
    basis_points = np.array([offset for c, offset in v if c == k])
    assert len(basis_points) >= 2
    assert (0, 0, 0) in [tuple(e) for e in basis_points]
    sizes = np.linalg.norm(basis_points, axis=1)
    index = np.argsort(sizes)[1]
    basis = basis_points[index]
    vector = np.dot(basis, atoms.get_cell())
    norm = np.linalg.norm(vector)
    vhat = vector / norm
    positions, numbers = build_supercomponent(atoms, components, k, v)
    scaled = np.dot(positions, orthogonal_basis(vhat).T / norm)
    scaled[:, 2] %= 1.0
    scaled[:, :2] -= np.mean(scaled, axis=0)[:2]
    scaled = select_chain_rotation(scaled)
    init_cell = norm * np.eye(3)
    pos = np.dot(scaled, init_cell)
    rmax = np.max(np.linalg.norm(pos[:, :2], axis=1))
    rmax = max(1, rmax)
    cell = np.diag([4 * rmax, 4 * rmax, norm])
    return Atoms(numbers=numbers, positions=pos, cell=cell, pbc=[0, 0, 1])