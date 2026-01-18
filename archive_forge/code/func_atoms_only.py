from __future__ import annotations
import abc
from collections import namedtuple
from collections.abc import Iterable
from enum import Enum, unique
from pprint import pformat
from typing import TYPE_CHECKING, cast
import numpy as np
from monty.collections import AttrDict
from monty.design_patterns import singleton
from monty.json import MontyDecoder, MontyEncoder, MSONable
from pymatgen.core import ArrayWithUnit, Lattice, Species, Structure, units
@classmethod
def atoms_only(cls, atoms_constraints=None):
    """Relax atomic positions, keep unit cell fixed."""
    if atoms_constraints is None:
        return cls(ionmov=cls.IONMOV_DEFAULT, optcell=0)
    return cls(ionmov=cls.IONMOV_DEFAULT, optcell=0, atoms_constraints=atoms_constraints)