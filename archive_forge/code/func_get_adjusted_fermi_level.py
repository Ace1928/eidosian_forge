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
def get_adjusted_fermi_level(efermi, cbm, band_structure):
    """
    When running a band structure computations the Fermi level needs to be
    take from the static run that gave the charge density used for the non-self
    consistent band structure run. Sometimes this Fermi level is however a
    little too low because of the mismatch between the uniform grid used in
    the static run and the band structure k-points (e.g., the VBM is on Gamma
    and the Gamma point is not in the uniform mesh). Here we use a procedure
    consisting in looking for energy levels higher than the static Fermi level
    (but lower than the LUMO) if any of these levels make the band structure
    appears insulating and not metallic anymore, we keep this adjusted fermi
    level. This procedure has shown to detect correctly most insulators.

    Args:
        efermi (float): The Fermi energy of the static run.
        cbm (float): The conduction band minimum of the static run.
        band_structure (BandStructureSymmLine): A band structure object.

    Returns:
        float: A new adjusted Fermi level.
    """
    bs_working = BandStructureSymmLine.from_dict(band_structure.as_dict())
    if bs_working.is_metal():
        energy = efermi
        while energy < cbm:
            energy += 0.01
            bs_working._efermi = energy
            if not bs_working.is_metal():
                return energy
    return efermi