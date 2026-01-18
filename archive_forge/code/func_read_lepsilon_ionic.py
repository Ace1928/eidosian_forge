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
def read_lepsilon_ionic(self):
    """
        Reads an LEPSILON run, the ionic component.

        # TODO: Document the actual variables.
        """
    try:
        search = []

        def dielectric_section_start(results, _match):
            results.dielectric_ionic_index = -1
        search.append(['MACROSCOPIC STATIC DIELECTRIC TENSOR IONIC', None, dielectric_section_start])

        def dielectric_section_start2(results, _match):
            results.dielectric_ionic_index = 0
        search.append(['-------------------------------------', lambda results, _line: results.dielectric_ionic_index == -1 if results.dielectric_ionic_index is not None else results.dielectric_ionic_index, dielectric_section_start2])

        def dielectric_data(results, match):
            results.dielectric_ionic_tensor[results.dielectric_ionic_index, :] = np.array([float(match.group(i)) for i in range(1, 4)])
            results.dielectric_ionic_index += 1
        search.append(['^ *([-0-9.Ee+]+) +([-0-9.Ee+]+) +([-0-9.Ee+]+) *$', lambda results, _line: results.dielectric_ionic_index >= 0 if results.dielectric_ionic_index is not None else results.dielectric_ionic_index, dielectric_data])

        def dielectric_section_stop(results, _match):
            results.dielectric_ionic_index = None
        search.append(['-------------------------------------', lambda results, _line: results.dielectric_ionic_index >= 1 if results.dielectric_ionic_index is not None else results.dielectric_ionic_index, dielectric_section_stop])
        self.dielectric_ionic_index = None
        self.dielectric_ionic_tensor = np.zeros((3, 3))

        def piezo_section_start(results, _match):
            results.piezo_ionic_index = 0
        search.append(['PIEZOELECTRIC TENSOR IONIC CONTR  for field in x, y, z        ', None, piezo_section_start])

        def piezo_data(results, match):
            results.piezo_ionic_tensor[results.piezo_ionic_index, :] = np.array([float(match.group(i)) for i in range(1, 7)])
            results.piezo_ionic_index += 1
        search.append(['^ *[xyz] +([-0-9.Ee+]+) +([-0-9.Ee+]+) +([-0-9.Ee+]+) *([-0-9.Ee+]+) +([-0-9.Ee+]+) +([-0-9.Ee+]+)*$', lambda results, _line: results.piezo_ionic_index >= 0 if results.piezo_ionic_index is not None else results.piezo_ionic_index, piezo_data])

        def piezo_section_stop(results, _match):
            results.piezo_ionic_index = None
        search.append(['-------------------------------------', lambda results, _line: results.piezo_ionic_index >= 1 if results.piezo_ionic_index is not None else results.piezo_ionic_index, piezo_section_stop])
        self.piezo_ionic_index = None
        self.piezo_ionic_tensor = np.zeros((3, 6))
        micro_pyawk(self.filename, search, self)
        self.dielectric_ionic_tensor = self.dielectric_ionic_tensor.tolist()
        self.piezo_ionic_tensor = self.piezo_ionic_tensor.tolist()
    except Exception:
        raise RuntimeError('ionic part of LEPSILON OUTCAR could not be parsed.')