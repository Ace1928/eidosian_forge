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
def read_igpar(self):
    """
        Renders accessible:
            er_ev = e<r>_ev (dictionary with Spin.up/Spin.down as keys)
            er_bp = e<r>_bp (dictionary with Spin.up/Spin.down as keys)
            er_ev_tot = spin up + spin down summed
            er_bp_tot = spin up + spin down summed
            p_elc = spin up + spin down summed
            p_ion = spin up + spin down summed.

        (See VASP section "LBERRY,  IGPAR,  NPPSTR,  DIPOL tags" for info on
        what these are).
        """
    self.er_ev = {}
    self.er_bp = {}
    self.er_ev_tot = None
    self.er_bp_tot = None
    self.p_elec = self.p_ion = None
    try:
        search = []

        def er_ev(results, match):
            results.er_ev[Spin.up] = np.array(map(float, match.groups()[1:4])) / 2
            results.er_ev[Spin.down] = results.er_ev[Spin.up]
            results.context = 2
        search.append(['^ *e<r>_ev=\\( *([-0-9.Ee+]*) *([-0-9.Ee+]*) *([-0-9.Ee+]*) *\\)', None, er_ev])

        def er_bp(results, match):
            results.er_bp[Spin.up] = np.array([float(match.group(i)) for i in range(1, 4)]) / 2
            results.er_bp[Spin.down] = results.er_bp[Spin.up]
        search.append(['^ *e<r>_bp=\\( *([-0-9.Ee+]*) *([-0-9.Ee+]*) *([-0-9.Ee+]*) *\\)', lambda results, _line: results.context == 2, er_bp])

        def er_ev_up(results, match):
            results.er_ev[Spin.up] = np.array([float(match.group(i)) for i in range(1, 4)])
            results.context = Spin.up
        search.append(['^.*Spin component 1 *e<r>_ev=\\( *([-0-9.Ee+]*) *([-0-9.Ee+]*) *([-0-9.Ee+]*) *\\)', None, er_ev_up])

        def er_bp_up(results, match):
            results.er_bp[Spin.up] = np.array([float(match.group(1)), float(match.group(2)), float(match.group(3))])
        search.append(['^ *e<r>_bp=\\( *([-0-9.Ee+]*) *([-0-9.Ee+]*) *([-0-9.Ee+]*) *\\)', lambda results, _line: results.context == Spin.up, er_bp_up])

        def er_ev_dn(results, match):
            results.er_ev[Spin.down] = np.array([float(match.group(1)), float(match.group(2)), float(match.group(3))])
            results.context = Spin.down
        search.append(['^.*Spin component 2 *e<r>_ev=\\( *([-0-9.Ee+]*) *([-0-9.Ee+]*) *([-0-9.Ee+]*) *\\)', None, er_ev_dn])

        def er_bp_dn(results, match):
            results.er_bp[Spin.down] = np.array([float(match.group(i)) for i in range(1, 4)])
        search.append(['^ *e<r>_bp=\\( *([-0-9.Ee+]*) *([-0-9.Ee+]*) *([-0-9.Ee+]*) *\\)', lambda results, _line: results.context == Spin.down, er_bp_dn])

        def p_elc(results, match):
            results.p_elc = np.array([float(match.group(i)) for i in range(1, 4)])
        search.append(['^.*Total electronic dipole moment: *p\\[elc\\]=\\( *([-0-9.Ee+]*) *([-0-9.Ee+]*) *([-0-9.Ee+]*) *\\)', None, p_elc])

        def p_ion(results, match):
            results.p_ion = np.array([float(match.group(i)) for i in range(1, 4)])
        search.append(['^.*ionic dipole moment: *p\\[ion\\]=\\( *([-0-9.Ee+]*) *([-0-9.Ee+]*) *([-0-9.Ee+]*) *\\)', None, p_ion])
        self.context = None
        self.er_ev = {Spin.up: None, Spin.down: None}
        self.er_bp = {Spin.up: None, Spin.down: None}
        micro_pyawk(self.filename, search, self)
        if self.er_ev[Spin.up] is not None and self.er_ev[Spin.down] is not None:
            self.er_ev_tot = self.er_ev[Spin.up] + self.er_ev[Spin.down]
        if self.er_bp[Spin.up] is not None and self.er_bp[Spin.down] is not None:
            self.er_bp_tot = self.er_bp[Spin.up] + self.er_bp[Spin.down]
    except Exception:
        raise RuntimeError('IGPAR OUTCAR could not be parsed.')