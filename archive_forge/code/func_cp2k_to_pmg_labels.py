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
def cp2k_to_pmg_labels(label: str) -> str:
    if label == 'p':
        return 'px'
    if label == 'd':
        return 'dxy'
    if label == 'f':
        return 'f_3'
    if label == 'd-2':
        return 'dxy'
    if label == 'd-1':
        return 'dyz'
    if label == 'd0':
        return 'dz2'
    if label == 'd+1':
        return 'dxz'
    if label == 'd+2':
        return 'dx2'
    if label == 'f-3':
        return 'f_3'
    if label == 'f-2':
        return 'f_2'
    if label == 'f-1':
        return 'f_1'
    if label == 'f0':
        return 'f0'
    if label == 'f+1':
        return 'f1'
    if label == 'f+2':
        return 'f2'
    if label == 'f+3':
        return 'f3'
    return label