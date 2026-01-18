from __future__ import annotations
import re
import warnings
from typing import TYPE_CHECKING
import numpy as np
import scipy.constants as cst
from monty.io import zopen
from scipy.stats import norm
from pymatgen.core import Composition, Element, Molecule
from pymatgen.core.operations import SymmOp
from pymatgen.core.units import Ha_to_eV
from pymatgen.electronic_structure.core import Spin
from pymatgen.util.coord import get_angle
from pymatgen.util.plotting import pretty_plot
def get_cart_coords(self) -> str:
    """Return the Cartesian coordinates of the molecule."""
    outs = []
    for site in self._mol:
        outs.append(f'{site.species_string} {' '.join((f'{x:0.6f}' for x in site.coords))}')
    return '\n'.join(outs)