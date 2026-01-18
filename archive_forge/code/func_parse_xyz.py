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
@classmethod
def parse_xyz(cls, filename: str | Path) -> pd.DataFrame:
    """
        Load xyz file generated from packmol (for those who find it hard to install openbabel).

        Returns:
            pandas.DataFrame
        """
    with zopen(filename, mode='rt') as file:
        lines = file.readlines()
    sio = StringIO(''.join(lines[2:]))
    df = pd.read_csv(sio, header=None, comment='#', delim_whitespace=True, names=['atom', 'x', 'y', 'z'])
    df.index += 1
    return df