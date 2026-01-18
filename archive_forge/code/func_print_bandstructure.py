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
def print_bandstructure(self, kpoints_line_density: int=20) -> None:
    """
        Attaches a non-scf band structure calc the end of an SCF loop.

        This requires a kpoint calculation, which is not always default in cp2k.

        Args:
            kpoints_line_density: number of kpoints along each branch in line-mode calc.
        """
    if not self.kpoints:
        raise ValueError('Kpoints must be provided to enable band structure printing')
    bs = BandStructure.from_kpoints(self.kpoints, kpoints_line_density=kpoints_line_density)
    self['force_eval']['dft']['print'].insert(bs)