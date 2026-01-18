from __future__ import annotations
import warnings
from typing import TYPE_CHECKING
import numpy as np
def get_grid_point_from_address(grid_address, mesh):
    """Return grid point index by translating grid address."""
    _set_no_error()
    return _spglib.grid_point_from_address(np.array(grid_address, dtype='intc'), np.array(mesh, dtype='intc'))