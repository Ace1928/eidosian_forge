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
def read_onsite_density_matrices(self):
    """
        Parse the onsite density matrices, returns list with index corresponding
        to atom index in Structure.
        """
    header_pattern = 'spin component  1\\n'
    row_pattern = '[^\\S\\r\\n]*(?:(-?[\\d.]+))' + '(?:[^\\S\\r\\n]*(-?[\\d.]+)[^\\S\\r\\n]*)?' * 6 + '.*?'
    footer_pattern = '\\nspin component  2'
    spin1_component = self.read_table_pattern(header_pattern, row_pattern, footer_pattern, postprocess=lambda x: float(x) if x else None, last_one_only=False)
    spin1_component = [[[e for e in row if e is not None] for row in matrix] for matrix in spin1_component]
    header_pattern = 'spin component  2\\n'
    row_pattern = '[^\\S\\r\\n]*(?:([\\d.-]+))' + '(?:[^\\S\\r\\n]*(-?[\\d.]+)[^\\S\\r\\n]*)?' * 6 + '.*?'
    footer_pattern = '\\n occupancies and eigenvectors'
    spin2_component = self.read_table_pattern(header_pattern, row_pattern, footer_pattern, postprocess=lambda x: float(x) if x else None, last_one_only=False)
    spin2_component = [[[e for e in row if e is not None] for row in matrix] for matrix in spin2_component]
    self.data['onsite_density_matrices'] = [{Spin.up: spin1_component[idx], Spin.down: spin2_component[idx]} for idx in range(len(spin1_component))]