from __future__ import annotations
from collections import namedtuple
from typing import TYPE_CHECKING
import numpy as np
from pymatgen.core import Site, Species
from pymatgen.core.tensors import SquareTensor
from pymatgen.core.units import FloatWithUnit
from pymatgen.util.due import Doi, due
@property
def asymmetry(self):
    """
        Asymmetry of the electric field tensor defined as:
            (V_yy - V_xx)/V_zz.
        """
    diags = np.diag(self.principal_axis_system)
    V = sorted(diags, key=np.abs)
    return np.abs((V[1] - V[0]) / V[2])