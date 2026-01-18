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
def fft_mesh(self, kpoint: int, band: int, spin: int=0, spinor: int=0, shift: bool=True) -> np.ndarray:
    """
        Places the coefficients of a wavefunction onto an fft mesh.

        Once the mesh has been obtained, a discrete fourier transform can be
        used to obtain real-space evaluation of the wavefunction. The output
        of this function can be passed directly to numpy's fft function. For
        example:

            mesh = Wavecar('WAVECAR').fft_mesh(kpoint, band)
            evals = np.fft.ifftn(mesh)

        Args:
            kpoint (int): the index of the kpoint where the wavefunction will be evaluated
            band (int): the index of the band where the wavefunction will be evaluated
            spin (int): the spin of the wavefunction for the desired
                wavefunction (only for ISPIN = 2, default = 0)
            spinor (int): component of the spinor that is evaluated (only used
                if vasp_type == 'ncl')
            shift (bool): determines if the zero frequency coefficient is
                placed at index (0, 0, 0) or centered

        Returns:
            a numpy ndarray representing the 3D mesh of coefficients
        """
    if self.vasp_type.lower()[0] == 'n':
        tcoeffs = self.coeffs[kpoint][band][spinor, :]
    elif self.spin == 2:
        tcoeffs = self.coeffs[spin][kpoint][band]
    else:
        tcoeffs = self.coeffs[kpoint][band]
    mesh = np.zeros(tuple(self.ng), dtype=np.complex128)
    for gp, coeff in zip(self.Gpoints[kpoint], tcoeffs):
        t = tuple(gp.astype(int) + (self.ng / 2).astype(int))
        mesh[t] = coeff
    if shift:
        return np.fft.ifftshift(mesh)
    return mesh