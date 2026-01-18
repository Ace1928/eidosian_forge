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
def as_ppmodel(cls, obj):
    """
        Constructs an instance of PPModel from obj.

        Accepts obj in the form:
            * PPmodel instance
            * string. e.g "godby:12.3 eV", "linden".
        """
    if isinstance(obj, cls):
        return obj
    if ':' not in obj:
        mode, plasmon_freq = (obj, None)
    else:
        mode, plasmon_freq = obj.split(':')
        try:
            plasmon_freq = float(plasmon_freq)
        except ValueError:
            plasmon_freq, unit = plasmon_freq.split()
            plasmon_freq = units.Energy(float(plasmon_freq), unit).to('Ha')
    return cls(mode=mode, plasmon_freq=plasmon_freq)