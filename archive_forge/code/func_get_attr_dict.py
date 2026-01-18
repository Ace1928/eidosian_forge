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
def get_attr_dict(self, raw=False, types=False):
    """Settings that go into .param file in a traditional dict"""
    attrdict = {k: o.raw_value if raw else o.value for k, o in self._options.items() if o.value is not None}
    if types:
        for key, val in attrdict.items():
            attrdict[key] = (val, self._options[key].type)
    return attrdict