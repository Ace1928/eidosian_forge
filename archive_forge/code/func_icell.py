import re
import warnings
from typing import Dict
import numpy as np
import ase  # Annotations
from ase.utils import jsonable
from ase.cell import Cell
@property
def icell(self) -> Cell:
    """Reciprocal cell of this BandPath as a :class:`~ase.cell.Cell`."""
    return self._icell