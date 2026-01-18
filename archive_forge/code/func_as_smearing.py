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
def as_smearing(cls, obj):
    """
        Constructs an instance of `Smearing` from obj. Accepts obj in the form:

            * Smearing instance
            * "name:tsmear"  e.g. "gaussian:0.004"  (Hartree units)
            * "name:tsmear units" e.g. "gaussian:0.1 eV"
            * None --> no smearing
        """
    if obj is None:
        return Smearing.nosmearing()
    if isinstance(obj, cls):
        return obj
    if obj == 'nosmearing':
        return cls.nosmearing()
    obj, tsmear = obj.split(':')
    obj.strip()
    occopt = cls._mode2occopt[obj]
    try:
        tsmear = float(tsmear)
    except ValueError:
        tsmear, unit = tsmear.split()
        tsmear = units.Energy(float(tsmear), unit).to('Ha')
    return cls(occopt, tsmear)