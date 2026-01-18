from __future__ import annotations
import copy
import logging
import math
import os
import re
import struct
import warnings
from typing import TYPE_CHECKING, Any
import networkx as nx
import numpy as np
import pandas as pd
from monty.io import zopen
from monty.json import MSONable, jsanitize
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.core import Molecule
from pymatgen.io.qchem.utils import (
def hessian_parser(filename: str='132.0', n_atoms: int | None=None) -> NDArray:
    """
    Parse the Hessian data from a Hessian scratch file.

    Args:
        filename: Path to the Hessian scratch file. Defaults to "132.0".
        n_atoms: Number of atoms in the molecule. If None, no reshaping will be done.

    Returns:
        NDArray: Hessian, formatted as 3n_atoms x 3n_atoms. Units are Hartree/Bohr^2/amu.
    """
    hessian: list[float] = []
    with zopen(filename, mode='rb') as file:
        binary = file.read()
    hessian.extend((struct.unpack('d', binary[ii * 8:(ii + 1) * 8])[0] for ii in range(len(binary) // 8)))
    if n_atoms:
        return np.reshape(hessian, (n_atoms * 3, n_atoms * 3))
    return np.array(hessian)