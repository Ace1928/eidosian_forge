from __future__ import annotations
import abc
import logging
import os
import sys
import warnings
from copy import deepcopy
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from monty.os.path import zpath
from monty.serialization import loadfn
from pymatgen.core.structure import Molecule, Structure
from pymatgen.io.feff.inputs import Atoms, Header, Potential, Tags
def all_input(self):
    """Returns all input files as a dict of {filename: feffio object}."""
    dct = {'HEADER': self.header(), 'PARAMETERS': self.tags}
    if 'RECIPROCAL' not in self.tags:
        dct.update({'POTENTIALS': self.potential, 'ATOMS': self.atoms})
    return dct