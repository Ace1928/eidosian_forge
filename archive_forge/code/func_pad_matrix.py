import sys
import copy
from typing import Union, NewType, Sequence, Tuple, Optional, Callable
def pad_matrix(self, matrix: Matrix, pad_value: int=0) -> Matrix:
    """
        Pad a possibly non-square matrix to make it square.

        **Parameters**

        - `matrix` (list of lists of numbers): matrix to pad
        - `pad_value` (`int`): value to use to pad the matrix

        **Returns**

        a new, possibly padded, matrix
        """
    max_columns = 0
    total_rows = len(matrix)
    for row in matrix:
        max_columns = max(max_columns, len(row))
    total_rows = max(max_columns, total_rows)
    new_matrix = []
    for row in matrix:
        row_len = len(row)
        new_row = row[:]
        if total_rows > row_len:
            new_row += [pad_value] * (total_rows - row_len)
        new_matrix += [new_row]
    while len(new_matrix) < total_rows:
        new_matrix += [[pad_value] * total_rows]
    return new_matrix