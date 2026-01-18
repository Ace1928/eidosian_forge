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
def get_band_structure_from_vasp_multiple_branches(dir_name, efermi=None, projections=False):
    """
    This method is used to get band structure info from a VASP directory. It
    takes into account that the run can be divided in several branches named
    "branch_x". If the run has not been divided in branches the method will
    turn to parsing vasprun.xml directly.

    The method returns None is there's a parsing error

    Args:
        dir_name: Directory containing all bandstructure runs.
        efermi: Efermi for bandstructure.
        projections: True if you want to get the data on site projections if
            any. Note that this is sometimes very large

    Returns:
        A BandStructure Object
    """
    if os.path.isfile(f'{dir_name}/branch_0'):
        branch_dir_names = [os.path.abspath(d) for d in glob(f'{dir_name}/branch_*') if os.path.isdir(d)]
        sorted_branch_dir_names = sorted(branch_dir_names, key=lambda x: int(x.split('_')[-1]))
        branches = []
        for dname in sorted_branch_dir_names:
            xml_file = f'{dname}/vasprun.xml'
            if os.path.isfile(xml_file):
                run = Vasprun(xml_file, parse_projected_eigen=projections)
                branches.append(run.get_band_structure(efermi=efermi))
            else:
                warnings.warn(f'Skipping {dname}. Unable to find {xml_file}')
        return get_reconstructed_band_structure(branches, efermi)
    xml_file = f'{dir_name}/vasprun.xml'
    if os.path.isfile(xml_file):
        return Vasprun(xml_file, parse_projected_eigen=projections).get_band_structure(kpoints_filename=None, efermi=efermi)
    return None