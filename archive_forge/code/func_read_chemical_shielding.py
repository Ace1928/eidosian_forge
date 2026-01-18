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
def read_chemical_shielding(self):
    """
        Parse the NMR chemical shieldings data. Only the second part "absolute, valence and core"
        will be parsed. And only the three right most field (ISO_SHIELDING, SPAN, SKEW) will be retrieved.

        Returns:
            List of chemical shieldings in the order of atoms from the OUTCAR. Maryland notation is adopted.
        """
    header_pattern = '\\s+CSA tensor \\(J\\. Mason, Solid State Nucl\\. Magn\\. Reson\\. 2, 285 \\(1993\\)\\)\\s+\\s+-{50,}\\s+\\s+EXCLUDING G=0 CONTRIBUTION\\s+INCLUDING G=0 CONTRIBUTION\\s+\\s+-{20,}\\s+-{20,}\\s+\\s+ATOM\\s+ISO_SHIFT\\s+SPAN\\s+SKEW\\s+ISO_SHIFT\\s+SPAN\\s+SKEW\\s+-{50,}\\s*$'
    first_part_pattern = '\\s+\\(absolute, valence only\\)\\s+$'
    swallon_valence_body_pattern = '.+?\\(absolute, valence and core\\)\\s+$'
    row_pattern = '\\d+(?:\\s+[-]?\\d+\\.\\d+){3}\\s+' + '\\s+'.join(['([-]?\\d+\\.\\d+)'] * 3)
    footer_pattern = '-{50,}\\s*$'
    h1 = header_pattern + first_part_pattern
    cs_valence_only = self.read_table_pattern(h1, row_pattern, footer_pattern, postprocess=float, last_one_only=True)
    h2 = header_pattern + swallon_valence_body_pattern
    cs_valence_and_core = self.read_table_pattern(h2, row_pattern, footer_pattern, postprocess=float, last_one_only=True)
    self.data['chemical_shielding'] = {'valence_only': cs_valence_only, 'valence_and_core': cs_valence_and_core}