from __future__ import annotations
import itertools
import re
import warnings
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal
import numpy as np
import pandas as pd
from monty.io import zopen
from monty.json import MSONable
from monty.serialization import loadfn
from ruamel.yaml import YAML
from pymatgen.core import Element, Lattice, Molecule, Structure
from pymatgen.core.operations import SymmOp
from pymatgen.util.io_utils import clean_lines
def as_lammpsdata(self):
    """
        Convert a CombinedData object to a LammpsData object. attributes are deep-copied.

        box (LammpsBox): Simulation box.
        force_fieldct (dict): Data for force field sections. Optional
            with default to None. Only keywords in force field and
            class 2 force field are valid keys, and each value is a
            DataFrame.
        topology (dict): Data for topology sections. Optional with
            default to None. Only keywords in topology are valid
            keys, and each value is a DataFrame.
        """
    items = {}
    items['box'] = LammpsBox(self.box.bounds, self.box.tilt)
    items['masses'] = self.masses.copy()
    items['atoms'] = self.atoms.copy()
    items['atom_style'] = self.atom_style
    items['velocities'] = None
    if self.force_field:
        all_ff_kws = SECTION_KEYWORDS['ff'] + SECTION_KEYWORDS['class2']
        items['force_field'] = {k: v.copy() for k, v in self.force_field.items() if k in all_ff_kws}
    if self.topology:
        items['topology'] = {k: v.copy() for k, v in self.topology.items() if k in SECTION_KEYWORDS['topology']}
    return LammpsData(**items)