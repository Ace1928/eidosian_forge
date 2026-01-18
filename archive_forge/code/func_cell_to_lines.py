import numpy as np
from math import sqrt
from itertools import islice
from ase.io.formats import string2index
from ase.utils import rotate
from ase.data import covalent_radii, atomic_numbers
from ase.data.colors import jmol_colors
def cell_to_lines(writer, cell):
    nlines = 0
    nsegments = []
    for c in range(3):
        d = sqrt((cell[c] ** 2).sum())
        n = max(2, int(d / 0.3))
        nsegments.append(n)
        nlines += 4 * n
    positions = np.empty((nlines, 3))
    T = np.empty(nlines, int)
    D = np.zeros((3, 3))
    n1 = 0
    for c in range(3):
        n = nsegments[c]
        dd = cell[c] / (4 * n - 2)
        D[c] = dd
        P = np.arange(1, 4 * n + 1, 4)[:, None] * dd
        T[n1:] = c
        for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            n2 = n1 + n
            positions[n1:n2] = P + i * cell[c - 2] + j * cell[c - 1]
            n1 = n2
    return (positions, T, D)