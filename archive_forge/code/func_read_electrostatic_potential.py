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
def read_electrostatic_potential(self):
    """Parses the eletrostatic potential for the last ionic step."""
    pattern = {'ngf': '\\s+dimension x,y,z NGXF=\\s+([\\.\\-\\d]+)\\sNGYF=\\s+([\\.\\-\\d]+)\\sNGZF=\\s+([\\.\\-\\d]+)'}
    self.read_pattern(pattern, postprocess=int)
    self.ngf = self.data.get('ngf', [[]])[0]
    pattern = {'radii': 'the test charge radii are((?:\\s+[\\.\\-\\d]+)+)'}
    self.read_pattern(pattern, reverse=True, terminate_on_match=True, postprocess=str)
    self.sampling_radii = [*map(float, self.data['radii'][0][0].split())]
    header_pattern = '\\(the norm of the test charge is\\s+[\\.\\-\\d]+\\)'
    table_pattern = '((?:\\s+\\d+\\s*[\\.\\-\\d]+)+)'
    footer_pattern = '\\s+E-fermi :'
    pots = self.read_table_pattern(header_pattern, table_pattern, footer_pattern)
    pots = ''.join(itertools.chain.from_iterable(pots))
    pots = re.findall('\\s+\\d+\\s*([\\.\\-\\d]+)+', pots)
    self.electrostatic_potential = [*map(float, pots)]