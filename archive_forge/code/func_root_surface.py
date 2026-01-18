from math import log10, atan2, cos, sin
from ase.build import hcp0001, fcc111, bcc111
import numpy as np
def root_surface(primitive_slab, root, eps=1e-08):
    """Creates a cell from a primitive cell that repeats along the x and y
    axis in a way consisent with the primitive cell, that has been cut
    to have a side length of *root*.

    *primitive cell* should be a primitive 2d cell of your slab, repeated
    as needed in the z direction.

    *root* should be determined using an analysis tool such as the
    root_surface_analysis function, or prior knowledge. It should always
    be a whole number as it represents the number of repetitions."""
    atoms = primitive_slab.copy()
    xscale, cell_vectors = _root_cell_normalization(primitive_slab)
    cell_points, root_point, roots = _root_surface_analysis(primitive_slab, root, eps=eps)
    root_angle = -atan2(root_point[1], root_point[0])
    root_rotation = [[cos(root_angle), -sin(root_angle)], [sin(root_angle), cos(root_angle)]]
    root_scale = np.linalg.norm(root_point)
    cell = np.array([np.dot(x, root_rotation) * root_scale for x in cell_vectors])
    shift = cell_vectors.sum(axis=0) / 2
    cell_points = [point for point in cell_points if point_in_cell_2d(point + shift, cell, eps=eps)]
    atoms.rotate(root_angle, v='z')
    atoms *= (root, root, 1)
    atoms.cell[0:2, 0:2] = cell * xscale
    atoms.center()
    del atoms[[atom.index for atom in atoms if not point_in_cell_2d(atom.position, atoms.cell, eps=eps)]]
    standard_rotation = [[cos(-root_angle), -sin(-root_angle), 0], [sin(-root_angle), cos(-root_angle), 0], [0, 0, 1]]
    new_cell = np.array([np.dot(x, standard_rotation) for x in atoms.cell])
    new_positions = np.array([np.dot(x, standard_rotation) for x in atoms.positions])
    atoms.cell = new_cell
    atoms.positions = new_positions
    return atoms