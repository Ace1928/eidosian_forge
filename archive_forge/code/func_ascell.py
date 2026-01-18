import ase
from typing import Mapping, Sequence, Union
import numpy as np
from ase.utils.arraywrapper import arraylike
from ase.utils import pbc2pbc
@classmethod
def ascell(cls, cell):
    """Return argument as a Cell object.  See :meth:`ase.cell.Cell.new`.

        A new Cell object is created if necessary."""
    if isinstance(cell, cls):
        return cell
    return cls.new(cell)