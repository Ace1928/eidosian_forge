from __future__ import annotations
import copy
import linecache
from io import StringIO
from typing import TYPE_CHECKING
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from pymatgen.io.pwmat.inputs import ACstrExtractor, AtomConfig, LineLocator
@property
def kpoints(self) -> np.ndarray:
    """Returns the fractional coordinates of kpoints."""
    return self._kpts