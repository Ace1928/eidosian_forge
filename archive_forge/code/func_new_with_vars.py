from __future__ import annotations
import abc
import copy
import json
import logging
import os
from collections import namedtuple
from collections.abc import Mapping, MutableMapping, Sequence
from enum import Enum, unique
from typing import TYPE_CHECKING
import numpy as np
from monty.collections import AttrDict
from monty.json import MSONable
from pymatgen.core.structure import Structure
from pymatgen.io.abinit import abiobjects as aobj
from pymatgen.io.abinit.pseudos import Pseudo, PseudoTable
from pymatgen.io.abinit.variable import InputVariable
from pymatgen.symmetry.bandstructure import HighSymmKpath
def new_with_vars(self, *args, **kwargs):
    """
        Return a new input with the given variables.

        Example:
            new = input.new_with_vars(ecut=20)
        """
    new = self.deepcopy()
    new.set_vars(*args, **kwargs)
    return new