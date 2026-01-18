import difflib
import numpy as np
import os
import re
import glob
import shutil
import sys
import json
import time
import tempfile
import warnings
import subprocess
from copy import deepcopy
from collections import namedtuple
from itertools import product
from typing import List, Set
import ase
import ase.units as units
from ase.calculators.general import Calculator
from ase.calculators.calculator import compare_atoms
from ase.calculators.calculator import PropertyNotImplementedError
from ase.calculators.calculator import kpts2sizeandoffsets
from ase.dft.kpoints import BandPath
from ase.parallel import paropen
from ase.io.castep import read_param
from ase.io.castep import read_bands
from ase.constraints import FixCartesian
def set_bandpath(self, bandpath):
    """Set a band structure path from ase.dft.kpoints.BandPath object

        This will set the bs_kpoint_list block with a set of specific points
        determined in ASE. bs_kpoint_spacing will not be used; to modify the
        number of points, consider using e.g. bandpath.resample(density=20) to
        obtain a new dense path.

        Args:
            bandpath (:obj:`ase.dft.kpoints.BandPath` or None):
                Set to None to remove list of band structure points. Otherwise,
                sampling will follow BandPath parameters.

        """

    def clear_bs_keywords():
        bs_keywords = product({'bs_kpoint', 'bs_kpoints'}, {'path', 'path_spacing', 'list', 'mp_grid', 'mp_spacing', 'mp_offset'})
        for bs_tag in bs_keywords:
            setattr(self.cell, '_'.join(bs_tag), None)
    if bandpath is None:
        clear_bs_keywords()
    elif isinstance(bandpath, BandPath):
        clear_bs_keywords()
        self.cell.bs_kpoint_list = [' '.join(map(str, row)) for row in bandpath.kpts]
    else:
        raise TypeError('Band structure path must be an ase.dft.kpoint.BandPath object')