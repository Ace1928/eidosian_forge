from __future__ import annotations
from math import isclose
from typing import TYPE_CHECKING
import numpy as np
import scipy.constants as const
from pymatgen.core.units import kb as kb_ev
from pymatgen.util.due import Doi, due
@classmethod
def from_qc_output(cls, output: QCOutput, **kwargs) -> Self:
    """
        Args:
            output (QCOutput): Pymatgen QCOutput object

        Returns:
            QuasiRRHO: QuasiRRHO class instantiated from a QChem Output
        """
    mult = output.data['multiplicity']
    elec_e = output.data['SCF_energy_in_the_final_basis_set']
    if output.data['optimization']:
        mol = output.data['molecule_from_last_geometry']
    else:
        mol = output.data['initial_molecule']
    frequencies = output.data['frequencies']
    return cls(mol=mol, frequencies=frequencies, energy=elec_e, mult=mult, **kwargs)