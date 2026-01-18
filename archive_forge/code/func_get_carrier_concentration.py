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
def get_carrier_concentration(self):
    """Gives the carrier concentration (in cm^-3).

        Returns:
            a dictionary {temp:[]} with an array of carrier concentration
            (in cm^-3) at each temperature
            The array relates to each step of electron chemical potential
        """
    return {temp: [1e+24 * i / self.vol for i in self._carrier_conc[temp]] for temp in self._carrier_conc}