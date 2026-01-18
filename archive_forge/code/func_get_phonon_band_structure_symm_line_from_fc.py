from __future__ import annotations
import numpy as np
from monty.dev import requires
from monty.serialization import loadfn
from scipy.interpolate import InterpolatedUnivariateSpline
from pymatgen.core import Lattice, Structure
from pymatgen.phonon.bandstructure import PhononBandStructure, PhononBandStructureSymmLine
from pymatgen.phonon.dos import CompletePhononDos, PhononDos
from pymatgen.phonon.gruneisen import GruneisenParameter, GruneisenPhononBandStructureSymmLine
from pymatgen.phonon.thermal_displacements import ThermalDisplacementMatrices
from pymatgen.symmetry.bandstructure import HighSymmKpath
@requires(Phonopy, 'phonopy is required to calculate phonon band structures')
def get_phonon_band_structure_symm_line_from_fc(structure: Structure, supercell_matrix: np.ndarray, force_constants: np.ndarray, line_density: float=20.0, symprec: float=0.01, **kwargs) -> PhononBandStructureSymmLine:
    """
    Get a phonon band structure along a high symmetry path from phonopy force
    constants.

    Args:
        structure: A structure.
        supercell_matrix: The supercell matrix used to generate the force
            constants.
        force_constants: The force constants in phonopy format.
        line_density: The density along the high symmetry path.
        symprec: Symmetry precision passed to phonopy and used for determining
            the band structure path.
        **kwargs: Additional kwargs passed to the Phonopy constructor.

    Returns:
        The line mode band structure.
    """
    structure_phonopy = get_phonopy_structure(structure)
    phonon = Phonopy(structure_phonopy, supercell_matrix=supercell_matrix, symprec=symprec, **kwargs)
    phonon.set_force_constants(force_constants)
    k_path = HighSymmKpath(structure, symprec=symprec)
    kpoints, labels = k_path.get_kpoints(line_density=line_density, coords_are_cartesian=False)
    phonon.run_qpoints(kpoints)
    frequencies = phonon.qpoints.get_frequencies().T
    labels_dict = {a: k for a, k in zip(labels, kpoints) if a != ''}
    return PhononBandStructureSymmLine(kpoints, frequencies, structure.lattice, labels_dict=labels_dict)