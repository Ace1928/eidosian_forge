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
def set_charge_atom_type(self, charges: dict[str | int, float]) -> None:
    """
        Add or modify charges of all atoms of a given type in the data.

        Args:
            charges: Dict containing the charges for the atom types to set.
                The dict should contain atom types as integers or labels and charges.
                Example: change the charge of Li atoms to +3:
                    charges={"Li": 3}
                    charges={1: 3} if Li atoms are of type 1
        """
    for iat, q in charges.items():
        if isinstance(iat, str):
            mass_iat = Element(iat).atomic_mass
            iat = self.masses.loc[self.masses['mass'] == mass_iat].index[0]
        self.atoms.loc[self.atoms['type'] == iat, 'q'] = q