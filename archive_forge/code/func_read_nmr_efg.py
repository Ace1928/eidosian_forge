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
def read_nmr_efg(self):
    """
        Parse the NMR Electric Field Gradient interpreted values.

        Returns:
            Electric Field Gradient tensors as a list of dict in the order of atoms from OUTCAR.
            Each dict key/value pair corresponds to a component of the tensors.
        """
    header_pattern = '^\\s+NMR quadrupolar parameters\\s+$\\n^\\s+Cq : quadrupolar parameter\\s+Cq=e[*]Q[*]V_zz/h$\\n^\\s+eta: asymmetry parameters\\s+\\(V_yy - V_xx\\)/ V_zz$\\n^\\s+Q  : nuclear electric quadrupole moment in mb \\(millibarn\\)$\\n^-{50,}$\\n^\\s+ion\\s+Cq\\(MHz\\)\\s+eta\\s+Q \\(mb\\)\\s+$\\n^-{50,}\\s*$\\n'
    row_pattern = '\\d+\\s+(?P<cq>[-]?\\d+\\.\\d+)\\s+(?P<eta>[-]?\\d+\\.\\d+)\\s+(?P<nuclear_quadrupole_moment>[-]?\\d+\\.\\d+)'
    footer_pattern = '-{50,}\\s*$'
    self.read_table_pattern(header_pattern, row_pattern, footer_pattern, postprocess=float, last_one_only=True, attribute_name='efg')