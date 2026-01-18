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
def save_scan_plot(self, filename='scan.pdf', img_format='pdf', coords=None):
    """
        Save matplotlib plot of the potential energy surface to a file.

        Args:
            filename: Filename to write to.
            img_format: Image format to use. Defaults to EPS.
            coords: internal coordinate name to use as abcissa.
        """
    plt = self.get_scan_plot(coords)
    plt.savefig(filename, format=img_format)