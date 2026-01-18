import numpy as np
from ase import Atoms
def get_deviation_from_optimal_cell_shape(cell, target_shape='sc', norm=None):
    """
    Calculates the deviation of the given cell metric from the ideal
    cell metric defining a certain shape. Specifically, the function
    evaluates the expression `\\Delta = || Q \\mathbf{h} -
    \\mathbf{h}_{target}||_2`, where `\\mathbf{h}` is the input
    metric (*cell*) and `Q` is a normalization factor (*norm*)
    while the target metric `\\mathbf{h}_{target}` (via
    *target_shape*) represent simple cubic ('sc') or face-centered
    cubic ('fcc') cell shapes.

    Parameters:

    cell: 2D array of floats
        Metric given as a (3x3 matrix) of the input structure.
    target_shape: str
        Desired supercell shape. Can be 'sc' for simple cubic or
        'fcc' for face-centered cubic.
    norm: float
        Specify the normalization factor. This is useful to avoid
        recomputing the normalization factor when computing the
        deviation for a series of P matrices.

    """
    if target_shape in ['sc', 'simple-cubic']:
        target_metric = np.eye(3)
    elif target_shape in ['fcc', 'face-centered cubic']:
        target_metric = 0.5 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    if not norm:
        norm = (np.linalg.det(cell) / np.linalg.det(target_metric)) ** (-1.0 / 3)
    return np.linalg.norm(norm * cell - target_metric)