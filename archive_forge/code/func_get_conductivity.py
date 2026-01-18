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
def get_conductivity(self, output='eigs', doping_levels=True, relaxation_time=1e-14):
    """Gives the conductivity (1/Ohm*m) in either a full 3x3 tensor
        form, as 3 eigenvalues, or as the average value
        (trace/3.0) If doping_levels=True, the results are given at
        different p and n doping
        levels (given by self.doping), otherwise it is given as a series
        of electron chemical potential values.

        Args:
            output (str): the type of output. 'tensor' give the full
            3x3 tensor, 'eigs' its 3 eigenvalues and
            'average' the average of the three eigenvalues
            doping_levels (bool): True for the results to be given at
            different doping levels, False for results
            at different electron chemical potentials
            relaxation_time (float): constant relaxation time in secs

        Returns:
            If doping_levels=True, a dictionary {temp:{'p':[],'n':[]}}.
            The 'p' links to conductivity
            at p-type doping and 'n' to the conductivity at n-type
            doping. Otherwise,
            returns a {temp:[]} dictionary. The result contains either
            the sorted three eigenvalues of the symmetric
            conductivity tensor (format='eigs') or a full tensor (3x3
            array) (output='tensor') or as an average
            (output='average').
            The result includes a given constant relaxation time

            units are 1/Ohm*m
        """
    return BoltztrapAnalyzer._format_to_output(self._cond, self._cond_doping, output, doping_levels, relaxation_time)