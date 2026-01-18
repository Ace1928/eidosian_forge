from __future__ import annotations
import itertools
import os
import warnings
import numpy as np
from ruamel.yaml import YAML
from pymatgen.core import SETTINGS
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Element, Molecule, Structure
from pymatgen.io.cp2k.inputs import (
from pymatgen.io.cp2k.utils import get_truncated_coulomb_cutoff, get_unique_site_indices
from pymatgen.io.vasp.inputs import Kpoints as VaspKpoints
from pymatgen.io.vasp.inputs import KpointsSupportedModes
@staticmethod
def get_cutoff_from_basis(basis_sets, rel_cutoff) -> float:
    """Given a basis and a relative cutoff. Determine the ideal cutoff variable."""
    for basis in basis_sets:
        if not basis.exponents:
            raise ValueError(f'Basis set {basis} contains missing exponent info. Please specify cutoff manually')

    def get_soft_exponents(b):
        if b.potential == 'All Electron':
            radius = 1.2 if b.element == Element('H') else 1.512
            threshold = 0.0001
            max_lshell = max((shell for shell in b.lmax))
            exponent = np.log(radius ** max_lshell / threshold) / radius ** 2
            return [[exponent]]
        return b.exponents
    exponents = [get_soft_exponents(b) for b in basis_sets if b.exponents]
    exponents = list(itertools.chain.from_iterable(exponents))
    return np.ceil(max(itertools.chain.from_iterable(exponents))) * rel_cutoff