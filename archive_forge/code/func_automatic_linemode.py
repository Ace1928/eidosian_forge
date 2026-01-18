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
def automatic_linemode(cls, divisions, ibz) -> Self:
    """
        Convenient static constructor for a KPOINTS in mode line_mode.
        gamma centered Monkhorst-Pack grids and the number of subdivisions
        along each reciprocal lattice vector determined by the scheme in the
        VASP manual.

        Args:
            divisions: Parameter determining the number of k-points along each high symmetry line.
            ibz: HighSymmKpath object (pymatgen.symmetry.bandstructure)

        Returns:
            Kpoints object
        """
    kpoints = []
    labels = []
    for path in ibz.kpath['path']:
        kpoints.append(ibz.kpath['kpoints'][path[0]])
        labels.append(path[0])
        for i in range(1, len(path) - 1):
            kpoints.append(ibz.kpath['kpoints'][path[i]])
            labels.append(path[i])
            kpoints.append(ibz.kpath['kpoints'][path[i]])
            labels.append(path[i])
        kpoints.append(ibz.kpath['kpoints'][path[-1]])
        labels.append(path[-1])
    return cls('Line_mode KPOINTS file', style=Kpoints.supported_modes.Line_mode, coord_type='Reciprocal', kpts=kpoints, labels=labels, num_kpts=int(divisions))