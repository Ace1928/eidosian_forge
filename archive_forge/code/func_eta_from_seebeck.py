from __future__ import annotations
import logging
import math
import os
import subprocess
import tempfile
import time
from shutil import which
from typing import TYPE_CHECKING, Literal
import numpy as np
from monty.dev import requires
from monty.json import MSONable, jsanitize
from monty.os import cd
from scipy import constants
from scipy.optimize import fsolve
from scipy.spatial import distance
from pymatgen.core.lattice import Lattice
from pymatgen.core.units import Energy, Length
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine, Kpoint
from pymatgen.electronic_structure.core import Orbital
from pymatgen.electronic_structure.dos import CompleteDos, Dos, Spin
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath
def eta_from_seebeck(seeb, Lambda):
    """It takes a value of seebeck and adjusts the analytic seebeck until it's equal.

    Returns:
        float: eta where the two seebeck coefficients are equal (reduced chemical potential).
    """
    out = fsolve(lambda x: (seebeck_spb(x, Lambda) - abs(seeb)) ** 2, 1.0, full_output=True)
    return out[0][0]