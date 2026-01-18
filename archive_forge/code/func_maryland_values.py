from __future__ import annotations
from collections import namedtuple
from typing import TYPE_CHECKING
import numpy as np
from pymatgen.core import Site, Species
from pymatgen.core.tensors import SquareTensor
from pymatgen.core.units import FloatWithUnit
from pymatgen.util.due import Doi, due
@property
def maryland_values(self):
    """Returns: the Chemical shielding tensor in Maryland Notation."""
    pas = self.principal_axis_system
    sigma_iso = pas.trace() / 3
    omega = np.diag(pas)[2] - np.diag(pas)[0]
    kappa = 3 * (np.diag(pas)[1] - sigma_iso) / omega
    return self.MarylandNotation(sigma_iso, omega, kappa)