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
def convert_str_to_int_float_or_str(value):
    """Safe eval()"""
    try:
        return int(value)
    except ValueError:
        try:
            value = float(value)
        except ValueError:
            value = {'True': True, 'False': False}.get(value, value)
        return value