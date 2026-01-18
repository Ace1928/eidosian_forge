from __future__ import annotations
import codecs
import contextlib
import hashlib
import itertools
import json
import logging
import math
import os
import re
import subprocess
import warnings
from collections import namedtuple
from enum import Enum, unique
from glob import glob
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Literal, cast
import numpy as np
import scipy.constants as const
from monty.io import zopen
from monty.json import MontyDecoder, MSONable
from monty.os import cd
from monty.os.path import zpath
from monty.serialization import dumpfn, loadfn
from tabulate import tabulate
from pymatgen.core import SETTINGS, Element, Lattice, Structure, get_el_sp
from pymatgen.electronic_structure.core import Magmom
from pymatgen.util.io_utils import clean_lines
from pymatgen.util.string import str_delimited
@classmethod
def automatic_gamma_density(cls, structure: Structure, kppa: float) -> Self:
    """
        Returns an automatic Kpoint object based on a structure and a kpoint
        density. Uses Gamma centered meshes always. For GW.

        Algorithm:
            Uses a simple approach scaling the number of divisions along each
            reciprocal lattice vector proportional to its length.

        Args:
            structure: Input structure
            kppa: Grid density
        """
    lattice = structure.lattice
    a, b, c = lattice.abc
    n_grid = kppa / len(structure)
    multip = (n_grid * a * b * c) ** (1 / 3)
    n_div = [int(round(multip / length)) for length in lattice.abc]
    n_div = [idx if idx > 0 else 1 for idx in n_div]
    n_div = [idx + idx % 2 if idx <= 8 else idx - idx % 2 + 1 for idx in n_div]
    style = Kpoints.supported_modes.Gamma
    comment = f'pymatgen with grid density = {kppa:.0f} / number of atoms'
    n_kpts = 0
    return cls(comment, n_kpts, style, [n_div], (0, 0, 0))