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
def get_castep_pp_path(castep_pp_path=''):
    """Abstract the quest for a CASTEP PSP directory."""
    if castep_pp_path:
        return os.path.abspath(os.path.expanduser(castep_pp_path))
    elif 'PSPOT_DIR' in os.environ:
        return os.environ['PSPOT_DIR']
    elif 'CASTEP_PP_PATH' in os.environ:
        return os.environ['CASTEP_PP_PATH']
    else:
        return os.path.abspath('.')