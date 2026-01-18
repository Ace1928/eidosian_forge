from __future__ import annotations
from collections import namedtuple
from typing import TYPE_CHECKING
import numpy as np
from pymatgen.core import Site, Species
from pymatgen.core.tensors import SquareTensor
from pymatgen.core.units import FloatWithUnit
from pymatgen.util.due import Doi, due
@property
def haeberlen_values(self):
    """Returns: the Chemical shielding tensor in Haeberlen Notation."""
    pas = self.principal_axis_system
    sigma_iso = pas.trace() / 3
    sigmas = np.diag(pas)
    sigmas = sorted(sigmas, key=lambda x: np.abs(x - sigma_iso))
    sigma_yy, sigma_xx, sigma_zz = sigmas
    delta_sigma = sigma_zz - 0.5 * (sigma_xx + sigma_yy)
    zeta = sigma_zz - sigma_iso
    eta = (sigma_yy - sigma_xx) / zeta
    return self.HaeberlenNotation(sigma_iso, delta_sigma, zeta, eta)