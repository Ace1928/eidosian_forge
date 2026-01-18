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
@property
def converged_ionic(self) -> bool:
    """
        Returns:
            bool: True if ionic step convergence has been reached, i.e. that vasp
                exited before reaching the max ionic steps for a relaxation run.
                In case IBRION=0 (MD) True if the max ionic steps are reached.
        """
    nsw = self.parameters.get('NSW', 0)
    ibrion = self.parameters.get('IBRION', -1 if nsw in (-1, 0) else 0)
    if ibrion == 0:
        return nsw <= 1 or self.md_n_steps == nsw
    return nsw <= 1 or len(self.ionic_steps) < nsw