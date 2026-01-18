from __future__ import annotations
import abc
import copy
import json
import logging
import os
from collections import namedtuple
from collections.abc import Mapping, MutableMapping, Sequence
from enum import Enum, unique
from typing import TYPE_CHECKING
import numpy as np
from monty.collections import AttrDict
from monty.json import MSONable
from pymatgen.core.structure import Structure
from pymatgen.io.abinit import abiobjects as aobj
from pymatgen.io.abinit.pseudos import Pseudo, PseudoTable
from pymatgen.io.abinit.variable import InputVariable
from pymatgen.symmetry.bandstructure import HighSymmKpath
def set_kpath(self, ndivsm, kptbounds=None, iscf=-2):
    """
        Set the variables for the computation of the electronic band structure.

        Args:
            ndivsm: Number of divisions for the smallest segment.
            kptbounds: k-points defining the path in k-space.
                If None, we use the default high-symmetry k-path defined in the pymatgen database.
        """
    if kptbounds is None:
        hsym_kpath = HighSymmKpath(self.structure)
        name2frac_coords = hsym_kpath.kpath['kpoints']
        kpath = hsym_kpath.kpath['path']
        frac_coords, names = ([], [])
        for segment in kpath:
            for name in segment:
                fc = name2frac_coords[name]
                frac_coords.append(fc)
                names.append(name)
        kptbounds = np.array(frac_coords)
    kptbounds = np.reshape(kptbounds, (-1, 3))
    return self.set_vars(kptbounds=kptbounds, kptopt=-(len(kptbounds) - 1), ndivsm=ndivsm, iscf=iscf)