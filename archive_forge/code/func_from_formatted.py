from __future__ import annotations
import datetime
import itertools
import logging
import math
import os
import re
import warnings
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from glob import glob
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal
import numpy as np
from monty.io import reverse_readfile, zopen
from monty.json import MSONable, jsanitize
from monty.os.path import zpath
from monty.re import regrep
from numpy.testing import assert_allclose
from pymatgen.core import Composition, Element, Lattice, Structure
from pymatgen.core.units import unitized
from pymatgen.electronic_structure.bandstructure import (
from pymatgen.electronic_structure.core import Magmom, Orbital, OrbitalType, Spin
from pymatgen.electronic_structure.dos import CompleteDos, Dos
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from pymatgen.io.common import VolumetricData as BaseVolumetricData
from pymatgen.io.core import ParseError
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, Potcar
from pymatgen.io.wannier90 import Unk
from pymatgen.util.io_utils import clean_lines, micro_pyawk
from pymatgen.util.num import make_symmetric_matrix_from_upper_tri
@classmethod
def from_formatted(cls, filename: str) -> Self:
    """Reads the WAVEDERF file and returns a Waveder object.

        Note: This file is only produced when LOPTICS is true AND vasp has been
        recompiled after uncommenting the line that calls
        WRT_CDER_BETWEEN_STATES_FORMATTED in linear_optics.F
        It is recommended to use `from_binary` instead since the binary file is
        much smaller and contains the same information.

        Args:
            filename (str): The name of the WAVEDER file.

        Returns:
            A Waveder object.
        """
    with zopen(filename, mode='rt') as file:
        nspin, nkpts, nbands = file.readline().split()
    data = np.loadtxt(filename, skiprows=1, usecols=(1, 4, 6, 7, 8, 9, 10, 11))
    data = data.reshape(int(nspin), int(nkpts), int(nbands), int(nbands), 8)
    cder_real = data[:, :, :, :, 2::2]
    cder_imag = data[:, :, :, :, 3::2]
    cder_real = np.transpose(cder_real, (2, 3, 1, 0, 4))
    cder_imag = np.transpose(cder_imag, (2, 3, 1, 0, 4))
    return cls(cder_real, cder_imag)