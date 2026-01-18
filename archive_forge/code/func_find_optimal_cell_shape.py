import numpy as np
from ase import Atoms
def find_optimal_cell_shape(cell, target_size, target_shape, lower_limit=-2, upper_limit=2, verbose=False):
    """Returns the transformation matrix that produces a supercell
    corresponding to *target_size* unit cells with metric *cell* that
    most closely approximates the shape defined by *target_shape*.

    Parameters:

    cell: 2D array of floats
        Metric given as a (3x3 matrix) of the input structure.
    target_size: integer
        Size of desired super cell in number of unit cells.
    target_shape: str
        Desired supercell shape. Can be 'sc' for simple cubic or
        'fcc' for face-centered cubic.
    lower_limit: int
        Lower limit of search range.
    upper_limit: int
        Upper limit of search range.
    verbose: bool
        Set to True to obtain additional information regarding
        construction of transformation matrix.

    """
    if target_shape in ['sc', 'simple-cubic']:
        target_metric = np.eye(3)
    elif target_shape in ['fcc', 'face-centered cubic']:
        target_metric = 0.5 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    if verbose:
        print('target metric (h_target):')
        print(target_metric)
    norm = (target_size * np.linalg.det(cell) / np.linalg.det(target_metric)) ** (-1.0 / 3)
    norm_cell = norm * cell
    if verbose:
        print('normalization factor (Q): %g' % norm)
    ideal_P = np.dot(target_metric, np.linalg.inv(norm_cell))
    if verbose:
        print('idealized transformation matrix:')
        print(ideal_P)
    starting_P = np.array(np.around(ideal_P, 0), dtype=int)
    if verbose:
        print('closest integer transformation matrix (P_0):')
        print(starting_P)
    from itertools import product
    best_score = 1000000.0
    optimal_P = None
    for dP in product(range(lower_limit, upper_limit + 1), repeat=9):
        dP = np.array(dP, dtype=int).reshape(3, 3)
        P = starting_P + dP
        if int(np.around(np.linalg.det(P), 0)) != target_size:
            continue
        score = get_deviation_from_optimal_cell_shape(np.dot(P, norm_cell), target_shape=target_shape, norm=1.0)
        if score < best_score:
            best_score = score
            optimal_P = P
    if optimal_P is None:
        print('Failed to find a transformation matrix.')
        return None
    if verbose:
        print('smallest score (|Q P h_p - h_target|_2): %f' % best_score)
        print('optimal transformation matrix (P_opt):')
        print(optimal_P)
        print('supercell metric:')
        print(np.round(np.dot(optimal_P, cell), 4))
        print('determinant of optimal transformation matrix: %g' % np.linalg.det(optimal_P))
    return optimal_P