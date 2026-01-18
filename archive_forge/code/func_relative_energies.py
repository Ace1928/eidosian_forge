from __future__ import annotations
import re
from collections import defaultdict
from typing import TYPE_CHECKING
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core import Element
from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.electronic_structure.dos import CompleteDos, Dos
from pymatgen.io.feff import Header, Potential, Tags
@property
def relative_energies(self):
    """
        Returns energy with respect to the Fermi level.
        E - E_f.
        """
    return self.data[:, 1]