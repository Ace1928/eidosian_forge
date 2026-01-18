from __future__ import annotations
import abc
import itertools
import os
import re
import shutil
import warnings
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from glob import glob
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Union, cast
from zipfile import ZipFile
import numpy as np
from monty.dev import deprecated
from monty.io import zopen
from monty.json import MSONable
from monty.serialization import loadfn
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Element, PeriodicSite, SiteCollection, Species, Structure
from pymatgen.io.core import InputGenerator
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, Potcar, VaspInput
from pymatgen.io.vasp.outputs import Outcar, Vasprun
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.util.due import Doi, due
def get_valid_magmom_struct(structure: Structure, inplace: bool=True, spin_mode: str='auto') -> Structure:
    """
    Make sure that the structure has valid magmoms based on the kind of calculation.

    Fill in missing Magmom values.

    Args:
        structure: The input structure
        inplace: True: edit magmoms of the input structure; False: return new structure
        spin_mode: "scalar"/"vector"/"none"/"auto" only first letter (s/v/n) is needed.
            dictates how the spin configuration will be determined.

            - auto: read the existing magmom values and decide
            - scalar: use a single scalar value (for spin up/down)
            - vector: use a vector value for spin-orbit systems
            - none: Remove all the magmom information

    Returns:
        New structure if inplace is False
    """
    default_values = {'s': 1.0, 'v': [1.0, 1.0, 1.0], 'n': None}
    if spin_mode[0].lower() == 'a':
        mode = 'n'
        for site in structure:
            if 'magmom' not in site.properties or site.properties['magmom'] is None:
                pass
            elif isinstance(site.properties['magmom'], (float, int)):
                if mode == 'v':
                    raise TypeError('Magmom type conflict')
                mode = 's'
                if isinstance(site.properties['magmom'], int):
                    site.properties['magmom'] = float(site.properties['magmom'])
            elif len(site.properties['magmom']) == 3:
                if mode == 's':
                    raise TypeError('Magmom type conflict')
                mode = 'v'
            else:
                raise TypeError('Unrecognized Magmom Value')
    else:
        mode = spin_mode[0].lower()
    ret_struct = structure if inplace else structure.copy()
    for site in ret_struct:
        if mode == 'n':
            if 'magmom' in site.properties:
                site.properties.pop('magmom')
        elif 'magmom' not in site.properties or site.properties['magmom'] is None:
            site.properties['magmom'] = default_values[mode]
    return ret_struct