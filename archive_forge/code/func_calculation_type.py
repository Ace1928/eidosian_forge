from __future__ import annotations
import logging
import os
import re
import warnings
from glob import glob
from itertools import chain
import numpy as np
import pandas as pd
from monty.io import zopen
from monty.json import MSONable, jsanitize
from monty.re import regrep
from pymatgen.core.structure import Molecule, Structure
from pymatgen.core.units import Ha_to_eV
from pymatgen.electronic_structure.bandstructure import BandStructure, BandStructureSymmLine
from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.electronic_structure.dos import CompleteDos, Dos
from pymatgen.io.cp2k.inputs import Keyword
from pymatgen.io.cp2k.sets import Cp2kInput
from pymatgen.io.cp2k.utils import natural_keys, postprocessor
from pymatgen.io.xyz import XYZ
@property
def calculation_type(self):
    """Returns the calculation type (what io.vasp.outputs calls run_type)."""
    LDA_TYPES = {'LDA', 'PADE', 'BECKE88', 'BECKE88_LR', 'BECKE88_LR_ADIABATIC', 'BECKE97'}
    GGA_TYPES = {'PBE', 'PW92'}
    HYBRID_TYPES = {'BLYP', 'B3LYP'}
    METAGGA_TYPES = {'TPSS': 'TPSS', 'RTPSS': 'revTPSS', 'M06L': 'M06-L', 'MBJ': 'modified Becke-Johnson', 'SCAN': 'SCAN', 'MS0': 'MadeSimple0', 'MS1': 'MadeSimple1', 'MS2': 'MadeSimple2'}
    functional = self.data.get('dft', {}).get('functional', [None])
    ip = self.data.get('dft', {}).get('hfx', {}).get('Interaction_Potential')
    frac = self.data.get('dft', {}).get('hfx', {}).get('FRACTION')
    if len(functional) > 1:
        rt = 'Mixed: ' + ', '.join(functional)
        functional = ' '.join(functional)
        if 'HYP' in functional or (ip and frac) or functional in HYBRID_TYPES:
            rt = 'Hybrid'
    else:
        functional = functional[0]
        if functional is None:
            rt = 'None'
        elif 'HYP' in functional or (ip and frac) or functional in HYBRID_TYPES:
            rt = 'Hybrid'
        elif 'MGGA' in functional or functional in METAGGA_TYPES:
            rt = 'METAGGA'
        elif 'GGA' in functional or functional in GGA_TYPES:
            rt = 'GGA'
        elif 'LDA' in functional or functional in LDA_TYPES:
            rt = 'LDA'
        else:
            rt = 'Unknown'
    if self.is_hubbard:
        rt += '+U'
    if self.data.get('dft').get('vdw'):
        rt += '+VDW'
    return rt