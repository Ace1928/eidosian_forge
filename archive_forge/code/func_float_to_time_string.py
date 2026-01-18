import functools
import json
import numbers
import operator
import os
import re
import warnings
from time import time
from typing import List, Dict, Any
import numpy as np
from ase.atoms import Atoms
from ase.calculators.calculator import all_properties, all_changes
from ase.data import atomic_numbers
from ase.db.row import AtomsRow
from ase.formula import Formula
from ase.io.jsonio import create_ase_object
from ase.parallel import world, DummyMPI, parallel_function, parallel_generator
from ase.utils import Lock, PurePath
def float_to_time_string(t, long=False):
    t *= YEAR
    for s in 'yMwdhms':
        x = t / seconds[s]
        if x > 5:
            break
    if long:
        return '{:.3f} {}s'.format(x, longwords[s])
    else:
        return '{:.0f}{}'.format(round(x), s)