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
def read_fermi_contact_shift(self):
    """
        Output example:
        Fermi contact (isotropic) hyperfine coupling parameter (MHz)
        -------------------------------------------------------------
        ion      A_pw      A_1PS     A_1AE     A_1c      A_tot
        -------------------------------------------------------------
         1      -0.002    -0.002    -0.051     0.000    -0.052
         2      -0.002    -0.002    -0.051     0.000    -0.052
         3       0.056     0.056     0.321    -0.048     0.321
        -------------------------------------------------------------
        , which corresponds to
        [[-0.002, -0.002, -0.051, 0.0, -0.052],
         [-0.002, -0.002, -0.051, 0.0, -0.052],
         [0.056, 0.056, 0.321, -0.048, 0.321]] from 'fch' data.
        """
    header_pattern1 = '\\s*Fermi contact \\(isotropic\\) hyperfine coupling parameter \\(MHz\\)\\s+\\s*\\-+\\s*ion\\s+A_pw\\s+A_1PS\\s+A_1AE\\s+A_1c\\s+A_tot\\s+\\s*\\-+'
    row_pattern1 = '(?:\\d+)\\s+' + '\\s+'.join(['([-]?\\d+\\.\\d+)'] * 5)
    footer_pattern = '\\-+'
    fch_table = self.read_table_pattern(header_pattern1, row_pattern1, footer_pattern, postprocess=float, last_one_only=True)
    header_pattern2 = '\\s*Dipolar hyperfine coupling parameters \\(MHz\\)\\s+\\s*\\-+\\s*ion\\s+A_xx\\s+A_yy\\s+A_zz\\s+A_xy\\s+A_xz\\s+A_yz\\s+\\s*\\-+'
    row_pattern2 = '(?:\\d+)\\s+' + '\\s+'.join(['([-]?\\d+\\.\\d+)'] * 6)
    dh_table = self.read_table_pattern(header_pattern2, row_pattern2, footer_pattern, postprocess=float, last_one_only=True)
    header_pattern3 = '\\s*Total hyperfine coupling parameters after diagonalization \\(MHz\\)\\s+\\s*\\(convention: \\|A_zz\\| > \\|A_xx\\| > \\|A_yy\\|\\)\\s+\\s*\\-+\\s*ion\\s+A_xx\\s+A_yy\\s+A_zz\\s+asymmetry \\(A_yy - A_xx\\)/ A_zz\\s+\\s*\\-+'
    row_pattern3 = '(?:\\d+)\\s+' + '\\s+'.join(['([-]?\\d+\\.\\d+)'] * 4)
    th_table = self.read_table_pattern(header_pattern3, row_pattern3, footer_pattern, postprocess=float, last_one_only=True)
    fc_shift_table = {'fch': fch_table, 'dh': dh_table, 'th': th_table}
    self.data['fermi_contact_shift'] = fc_shift_table