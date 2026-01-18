import os
import copy
import subprocess
from math import pi, sqrt
import pathlib
from typing import Union, Optional, List, Set, Dict, Any
import warnings
import numpy as np
from ase.cell import Cell
from ase.outputs import Properties, all_outputs
from ase.utils import jsonable
from ase.calculators.abc import GetPropertiesMixin
def register_calculator_class(name, cls):
    """ Add the class into the database. """
    assert name not in external_calculators
    external_calculators[name] = cls
    names.append(name)
    names.sort()