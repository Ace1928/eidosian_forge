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
def get_box_shift(self, i: Sequence[int]) -> np.ndarray:
    """
        Calculates the coordinate shift due to PBC.

        Args:
            i: A (n, 3) integer array containing the labels for box
            images of n entries.

        Returns:
            Coordinate shift array with the same shape of i
        """
    return np.inner(i, self._matrix.T)