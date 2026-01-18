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
def seebeck_spb(eta, Lambda=0.5):
    """Seebeck analytic formula in the single parabolic model."""
    try:
        from fdint import fdk
    except ImportError:
        raise BoltztrapError('fdint module not found. Please, install it.\nIt is needed to calculate Fermi integral quickly.')
    return constants.k / constants.e * ((2.0 + Lambda) * fdk(1.0 + Lambda, eta) / ((1.0 + Lambda) * fdk(Lambda, eta)) - eta) * 1000000.0