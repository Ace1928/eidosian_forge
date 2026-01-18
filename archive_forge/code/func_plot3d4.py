import fractions
import functools
import re
from collections import OrderedDict
from typing import List, Tuple, Dict
import numpy as np
from scipy.spatial import ConvexHull
import ase.units as units
from ase.formula import Formula
def plot3d4(self, ax):
    x, y, z = self.points[:, 1:-1].T
    a = x / 2 + y + z / 2
    b = 3 ** 0.5 * (x / 2 + y / 6)
    c = (2 / 3) ** 0.5 * z
    ax.scatter(a[self.hull], b[self.hull], c[self.hull], c='g', marker='o')
    ax.scatter(a[~self.hull], b[~self.hull], c[~self.hull], c='r', marker='s')
    for x, y, z, ref in zip(a, b, c, self.references):
        name = re.sub('(\\d+)', '$_{\\1}$', ref[2])
        ax.text(x, y, z, name, ha='center', va='bottom')
    for i, j, k, w in self.simplices:
        ax.plot(a[[i, j, k, i, w, k, j, w]], b[[i, j, k, i, w, k, j, w]], zs=c[[i, j, k, i, w, k, j, w]], c='b')
    ax.set_xlim3d(0, 1)
    ax.set_ylim3d(0, 1)
    ax.set_zlim3d(0, 1)
    ax.view_init(azim=115, elev=30)