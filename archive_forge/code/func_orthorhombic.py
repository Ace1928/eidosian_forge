import numpy as np
from numpy import pi, sin, cos, arccos, sqrt, dot
from numpy.linalg import norm
from ase.cell import Cell  # noqa
def orthorhombic(cell):
    """Return cell as three box dimensions or raise ValueError."""
    if not is_orthorhombic(cell):
        raise ValueError('Not orthorhombic')
    return cell.diagonal().copy()