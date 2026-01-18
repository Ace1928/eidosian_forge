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
def get_parchg(self, poscar: Poscar, kpoint: int, band: int, spin: int | None=None, spinor: int | None=None, phase: bool=False, scale: int=2) -> Chgcar:
    """
        Generates a Chgcar object, which is the charge density of the specified
        wavefunction.

        This function generates a Chgcar object with the charge density of the
        wavefunction specified by band and kpoint (and spin, if the WAVECAR
        corresponds to a spin-polarized calculation). The phase tag is a
        feature that is not present in VASP. For a real wavefunction, the phase
        tag being turned on means that the charge density is multiplied by the
        sign of the wavefunction at that point in space. A warning is generated
        if the phase tag is on and the chosen kpoint is not Gamma.

        Note: Augmentation from the PAWs is NOT included in this function. The
        maximal charge density will differ from the PARCHG from VASP, but the
        qualitative shape of the charge density will match.

        Args:
            poscar (pymatgen.io.vasp.inputs.Poscar): Poscar object that has the
                structure associated with the WAVECAR file
            kpoint (int): the index of the kpoint for the wavefunction
            band (int): the index of the band for the wavefunction
            spin (int): optional argument to specify the spin. If the Wavecar
                has ISPIN = 2, spin is None generates a Chgcar with total spin
                and magnetization, and spin == {0, 1} specifies just the spin
                up or down component.
            spinor (int): optional argument to specify the spinor component
                for noncollinear data wavefunctions (allowed values of None,
                0, or 1)
            phase (bool): flag to determine if the charge density is multiplied
                by the sign of the wavefunction. Only valid for real
                wavefunctions.
            scale (int): scaling for the FFT grid. The default value of 2 is at
                least as fine as the VASP default.

        Returns:
            a pymatgen.io.vasp.outputs.Chgcar object
        """
    if phase and (not np.all(self.kpoints[kpoint] == 0.0)):
        warnings.warn("phase is True should only be used for the Gamma kpoint! I hope you know what you're doing!")
    temp_ng = self.ng
    self.ng = self.ng * scale
    N = np.prod(self.ng)
    data = {}
    if self.spin == 2:
        if spin is not None:
            wfr = np.fft.ifftn(self.fft_mesh(kpoint, band, spin=spin)) * N
            den = np.abs(np.conj(wfr) * wfr)
            if phase:
                den = np.sign(np.real(wfr)) * den
            data['total'] = den
        else:
            wfr = np.fft.ifftn(self.fft_mesh(kpoint, band, spin=0)) * N
            denup = np.abs(np.conj(wfr) * wfr)
            wfr = np.fft.ifftn(self.fft_mesh(kpoint, band, spin=1)) * N
            dendn = np.abs(np.conj(wfr) * wfr)
            data['total'] = denup + dendn
            data['diff'] = denup - dendn
    else:
        if spinor is not None:
            wfr = np.fft.ifftn(self.fft_mesh(kpoint, band, spinor=spinor)) * N
            den = np.abs(np.conj(wfr) * wfr)
        else:
            wfr = np.fft.ifftn(self.fft_mesh(kpoint, band, spinor=0)) * N
            wfr_t = np.fft.ifftn(self.fft_mesh(kpoint, band, spinor=1)) * N
            den = np.abs(np.conj(wfr) * wfr)
            den += np.abs(np.conj(wfr_t) * wfr_t)
        if phase and (not (self.vasp_type.lower()[0] == 'n' and spinor is None)):
            den = np.sign(np.real(wfr)) * den
        data['total'] = den
    self.ng = temp_ng
    return Chgcar(poscar, data)