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
def e_fermi(self) -> float:
    """Returns the fermi energy level.

        Returns:
            float: Fermi energy level.
        """
    return self._e_fermi