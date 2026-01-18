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
def read_lcalcpol(self):
    """
        Reads the lcalpol.

        # TODO: Document the actual variables.
        """
    self.p_elec = self.p_sp1 = self.p_sp2 = self.p_ion = None
    try:
        search = []

        def p_elec(results, match):
            results.p_elec = np.array([float(match.group(1)), float(match.group(2)), float(match.group(3))])
        search.append(['^.*Total electronic dipole moment: *p\\[elc\\]=\\( *([-0-9.Ee+]*) *([-0-9.Ee+]*) *([-0-9.Ee+]*) *\\)', None, p_elec])
        if self.spin and (not self.noncollinear):

            def p_sp1(results, match):
                results.p_sp1 = np.array([float(match.group(1)), float(match.group(2)), float(match.group(3))])
            search.append(['^.*p\\[sp1\\]=\\( *([-0-9.Ee+]*) *([-0-9.Ee+]*) *([-0-9.Ee+]*) *\\)', None, p_sp1])

            def p_sp2(results, match):
                results.p_sp2 = np.array([float(match.group(1)), float(match.group(2)), float(match.group(3))])
            search.append(['^.*p\\[sp2\\]=\\( *([-0-9.Ee+]*) *([-0-9.Ee+]*) *([-0-9.Ee+]*) *\\)', None, p_sp2])

        def p_ion(results, match):
            results.p_ion = np.array([float(match.group(1)), float(match.group(2)), float(match.group(3))])
        search.append(['^.*Ionic dipole moment: *p\\[ion\\]=\\( *([-0-9.Ee+]*) *([-0-9.Ee+]*) *([-0-9.Ee+]*) *\\)', None, p_ion])
        micro_pyawk(self.filename, search, self)
        regex = '^.*Ionic dipole moment: .*'
        search = [[regex, None, lambda x, y: x.append(y.group(0))]]
        results = micro_pyawk(self.filename, search, [])
        if '|e|' in results[0]:
            self.p_elec *= -1
            self.p_ion *= -1
            if self.spin and (not self.noncollinear):
                self.p_sp1 *= -1
                self.p_sp2 *= -1
    except Exception as exc:
        print(exc.args)
        raise RuntimeError('LCALCPOL OUTCAR could not be parsed.') from exc