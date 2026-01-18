from math import cos, sin, sqrt
from os.path import basename
import numpy as np
from ase.calculators.calculator import PropertyNotImplementedError
from ase.data import atomic_numbers
from ase.data.colors import jmol_colors
from ase.geometry import complete_cell
from ase.gui.repeat import Repeat
from ase.gui.rotate import Rotate
from ase.gui.render import Render
from ase.gui.colors import ColorWindow
from ase.gui.utils import get_magmoms
from ase.utils import rotate
def get_cell_coordinates(cell, shifted=False):
    """Get start and end points of lines segments used to draw cell."""
    nn = []
    for c in range(3):
        v = cell[c]
        d = sqrt(np.dot(v, v))
        if d < 1e-12:
            n = 0
        else:
            n = max(2, int(d / 0.3))
        nn.append(n)
    B1 = np.zeros((2, 2, sum(nn), 3))
    B2 = np.zeros((2, 2, sum(nn), 3))
    n1 = 0
    for c, n in enumerate(nn):
        n2 = n1 + n
        h = 1.0 / (2 * n - 1)
        R = np.arange(n) * (2 * h)
        for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            B1[i, j, n1:n2, c] = R
            B1[i, j, n1:n2, (c + 1) % 3] = i
            B1[i, j, n1:n2, (c + 2) % 3] = j
        B2[:, :, n1:n2] = B1[:, :, n1:n2]
        B2[:, :, n1:n2, c] += h
        n1 = n2
    B1.shape = (-1, 3)
    B2.shape = (-1, 3)
    if shifted:
        B1 -= 0.5
        B2 -= 0.5
    return (B1, B2)