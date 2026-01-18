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
def read_cs_g0_contribution(self):
    """
        Parse the  G0 contribution of NMR chemical shielding.

        Returns:
            G0 contribution matrix as list of list.
        """
    header_pattern = '^\\s+G\\=0 CONTRIBUTION TO CHEMICAL SHIFT \\(field along BDIR\\)\\s+$\\n^\\s+-{50,}$\\n^\\s+BDIR\\s+X\\s+Y\\s+Z\\s*$\\n^\\s+-{50,}\\s*$\\n'
    row_pattern = '(?:\\d+)\\s+' + '\\s+'.join(['([-]?\\d+\\.\\d+)'] * 3)
    footer_pattern = '\\s+-{50,}\\s*$'
    self.read_table_pattern(header_pattern, row_pattern, footer_pattern, postprocess=float, last_one_only=True, attribute_name='cs_g0_contribution')