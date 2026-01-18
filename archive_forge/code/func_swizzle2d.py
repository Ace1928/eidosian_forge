from __future__ import annotations
from ..runtime.jit import jit
from . import core, math
@jit
def swizzle2d(i, j, size_i, size_j, size_g):
    """
    Transforms indices of a row-major size_i*size_j matrix into those
    of one where indices are row major for each group of size_j rows.
    For example, for size_i = size_j = 4 and size_g = 2, it will transform
    [[0 , 1 , 2 , 3 ],
     [4 , 5 , 6 , 7 ],
     [8 , 9 , 10, 11],
     [12, 13, 14, 15]]
    into
    [[0, 2,  4 , 6 ],
     [1, 3,  5 , 7 ],
     [8, 10, 12, 14],
     [9, 11, 13, 15]]
    """
    ij = i * size_j + j
    size_gj = size_g * size_j
    group_id = ij // size_gj
    off_i = group_id * size_g
    size_g = minimum(size_i - off_i, size_g)
    new_i = off_i + ij % size_g
    new_j = ij % size_gj // size_g
    return (new_i, new_j)