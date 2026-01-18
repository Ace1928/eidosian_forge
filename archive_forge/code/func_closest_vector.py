import itertools
import numpy as np
from ase.utils import pbc2pbc
from ase.cell import Cell
def closest_vector(t0, u, v):
    t = t0
    a = np.zeros(2, dtype=int)
    rs, cs = relevant_vectors_2D(u, v)
    dprev = float('inf')
    for it in range(MAX_IT):
        ds = np.linalg.norm(rs + t, axis=1)
        index = np.argmin(ds)
        if index == 0 or ds[index] >= dprev:
            return a
        dprev = ds[index]
        r = rs[index]
        kopt = int(round(-np.dot(t, r) / np.dot(r, r)))
        a += kopt * cs[index]
        t = t0 + a[0] * u + a[1] * v
    raise RuntimeError(f'Closest vector not found after {MAX_IT} iterations')