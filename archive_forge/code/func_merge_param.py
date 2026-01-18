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
def merge_param(self, param, overwrite=True, ignore_internal_keys=False):
    """Parse a param file and merge it into the current parameters."""
    if isinstance(param, CastepParam):
        for key, option in param._options.items():
            if option.value is not None:
                self.param.__setattr__(key, option.value)
        return
    elif isinstance(param, str):
        param_file = open(param, 'r')
        _close = True
    else:
        param_file = param
        attributes = ['name', 'closereadlines']
        for attr in attributes:
            if not hasattr(param_file, attr):
                raise TypeError('"param" is neither CastepParam nor str nor valid fileobj')
        param = param_file.name
        _close = False
    self, int_opts = read_param(fd=param_file, calc=self, get_interface_options=True)
    for k, val in int_opts.items():
        if k in self.internal_keys and (not ignore_internal_keys):
            if val in _tf_table:
                val = _tf_table[val]
            self._opt[k] = val
    if _close:
        param_file.close()