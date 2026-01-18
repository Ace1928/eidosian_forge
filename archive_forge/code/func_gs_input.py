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
def gs_input(structure, pseudos, kppa=None, ecut=None, pawecutdg=None, scf_nband=None, accuracy='normal', spin_mode='polarized', smearing='fermi_dirac:0.1 eV', charge=0.0, scf_algorithm=None):
    """
    Returns a |BasicAbinitInput| for ground-state calculation.

    Args:
        structure: |Structure| object.
        pseudos: List of filenames or list of |Pseudo| objects or |PseudoTable| object.
        kppa: Defines the sampling used for the SCF run. Defaults to 1000 if not given.
        ecut: cutoff energy in Ha (if None, ecut is initialized from the pseudos according to accuracy)
        pawecutdg: cutoff energy in Ha for PAW double-grid (if None, pawecutdg is initialized from the pseudos
            according to accuracy)
        scf_nband: Number of bands for SCF run. If scf_nband is None, nband is automatically initialized
            from the list of pseudos, the structure and the smearing option.
        accuracy: Accuracy of the calculation.
        spin_mode: Spin polarization.
        smearing: Smearing technique.
        charge: Electronic charge added to the unit cell.
        scf_algorithm: Algorithm used for solving of the SCF cycle.
    """
    multi = ebands_input(structure, pseudos, kppa=kppa, ndivsm=0, ecut=ecut, pawecutdg=pawecutdg, scf_nband=scf_nband, accuracy=accuracy, spin_mode=spin_mode, smearing=smearing, charge=charge, scf_algorithm=scf_algorithm)
    return multi[0]