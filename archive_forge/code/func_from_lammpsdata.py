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
def from_lammpsdata(cls, mols: list, names: list, list_of_numbers: list, coordinates: pd.DataFrame, atom_style: str | None=None) -> Self:
    """
        Constructor that can infer atom_style.
        The input LammpsData objects are used non-destructively.

        Args:
            mols: a list of LammpsData of a chemical cluster.Each LammpsData object (cluster)
                may contain one or more molecule ID.
            names: a list of name for each cluster.
            list_of_numbers: a list of Integer for counts of each molecule
            coordinates (pandas.DataFrame): DataFrame at least containing
                columns of ["x", "y", "z"] for coordinates of atoms.
            atom_style (str): Output atom_style. Default to "full".
        """
    styles = [mol.atom_style for mol in mols]
    if len(set(styles)) != 1:
        raise ValueError('Data have different atom_style.')
    style_return = styles.pop()
    if atom_style and atom_style != style_return:
        raise ValueError('Data have different atom_style as specified.')
    return cls(mols, names, list_of_numbers, coordinates, style_return)