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
def read_cs_raw_symmetrized_tensors(self):
    """
        Parse the matrix form of NMR tensor before corrected to table.

        Returns:
            nsymmetrized tensors list in the order of atoms.
        """
    header_pattern = '\\s+-{50,}\\s+\\s+Absolute Chemical Shift tensors\\s+\\s+-{50,}$'
    first_part_pattern = '\\s+UNSYMMETRIZED TENSORS\\s+$'
    row_pattern = '\\s+'.join(['([-]?\\d+\\.\\d+)'] * 3)
    unsym_footer_pattern = '^\\s+SYMMETRIZED TENSORS\\s+$'
    with zopen(self.filename, mode='rt') as file:
        text = file.read()
    unsym_table_pattern_text = header_pattern + first_part_pattern + '(?P<table_body>.+)' + unsym_footer_pattern
    table_pattern = re.compile(unsym_table_pattern_text, re.MULTILINE | re.DOTALL)
    rp = re.compile(row_pattern)
    m = table_pattern.search(text)
    if m:
        table_text = m.group('table_body')
        micro_header_pattern = 'ion\\s+\\d+'
        micro_table_pattern_text = micro_header_pattern + '\\s*^(?P<table_body>(?:\\s*' + row_pattern + ')+)\\s+'
        micro_table_pattern = re.compile(micro_table_pattern_text, re.MULTILINE | re.DOTALL)
        unsym_tensors = []
        for mt in micro_table_pattern.finditer(table_text):
            table_body_text = mt.group('table_body')
            tensor_matrix = []
            for line in table_body_text.rstrip().split('\n'):
                ml = rp.search(line)
                processed_line = [float(v) for v in ml.groups()]
                tensor_matrix.append(processed_line)
            unsym_tensors.append(tensor_matrix)
        self.data['unsym_cs_tensor'] = unsym_tensors
    else:
        raise ValueError('NMR UNSYMMETRIZED TENSORS is not found')