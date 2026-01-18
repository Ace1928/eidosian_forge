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
def lattice_2_lmpbox(lattice: Lattice, origin: Sequence=(0, 0, 0)) -> tuple[LammpsBox, SymmOp]:
    """
    Converts a lattice object to LammpsBox, and calculates the symmetry
    operation used.

    Args:
        lattice (Lattice): Input lattice.
        origin: A (3,) array/list of floats setting lower bounds of
            simulation box. Default to (0, 0, 0).

    Returns:
        LammpsBox, SymmOp
    """
    a, b, c = lattice.abc
    xlo, ylo, zlo = origin
    xhi = a + xlo
    m = lattice.matrix
    xy = np.dot(m[1], m[0] / a)
    yhi = np.sqrt(b ** 2 - xy ** 2) + ylo
    xz = np.dot(m[2], m[0] / a)
    yz = (np.dot(m[1], m[2]) - xy * xz) / (yhi - ylo)
    zhi = np.sqrt(c ** 2 - xz ** 2 - yz ** 2) + zlo
    tilt = None if lattice.is_orthogonal else [xy, xz, yz]
    rot_matrix = np.linalg.solve([[xhi - xlo, 0, 0], [xy, yhi - ylo, 0], [xz, yz, zhi - zlo]], m)
    bounds = [[xlo, xhi], [ylo, yhi], [zlo, zhi]]
    symm_op = SymmOp.from_rotation_and_translation(rot_matrix, origin)
    return (LammpsBox(bounds, tilt), symm_op)