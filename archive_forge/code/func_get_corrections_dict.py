from __future__ import annotations
import abc
import copy
import os
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Literal, Union
import numpy as np
from monty.design_patterns import cached_class
from monty.json import MSONable
from monty.serialization import loadfn
from tqdm import tqdm
from uncertainties import ufloat
from pymatgen.analysis.structure_analyzer import oxide_type, sulfide_type
from pymatgen.core import SETTINGS, Composition, Element
from pymatgen.entries.computed_entries import (
from pymatgen.io.vasp.sets import MITRelaxSet, MPRelaxSet, VaspInputSet
from pymatgen.util.due import Doi, due
def get_corrections_dict(self, entry: AnyComputedEntry) -> tuple[dict[str, float], dict[str, float]]:
    """Returns the correction values and uncertainties applied to a particular entry.

        Args:
            entry: A ComputedEntry object.

        Returns:
            tuple[dict[str, float], dict[str, float]]: Map from correction names to values
                (1st) and uncertainties (2nd).
        """
    corrections = {}
    uncertainties = {}
    for c in self.corrections:
        val = c.get_correction(entry)
        if val != 0:
            corrections[str(c)] = val.nominal_value
            uncertainties[str(c)] = val.std_dev
    return (corrections, uncertainties)