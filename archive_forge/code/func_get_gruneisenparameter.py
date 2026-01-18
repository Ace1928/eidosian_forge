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
def get_gruneisenparameter(gruneisen_path, structure=None, structure_path=None) -> GruneisenParameter:
    """
    Get Gruneisen object from gruneisen.yaml file, as obtained from phonopy (Frequencies in THz!).
    The order is structure > structure path > structure from gruneisen dict.
    Newer versions of phonopy include the structure in the yaml file,
    the structure/structure_path is kept for compatibility.

    Args:
        gruneisen_path: Path to gruneisen.yaml file (frequencies have to be in THz!)
        structure: pymatgen Structure object
        structure_path: path to structure in a file (e.g., POSCAR)

    Returns:
        GruneisenParameter
    """
    gruneisen_dict = loadfn(gruneisen_path)
    if structure_path and structure is None:
        structure = Structure.from_file(structure_path)
    else:
        try:
            structure = get_structure_from_dict(gruneisen_dict)
        except ValueError as exc:
            raise ValueError('Please provide a structure or structure path') from exc
    q_pts, multiplicities, frequencies, gruneisen = ([] for _ in range(4))
    phonopy_labels_dict = {}
    for p in gruneisen_dict['phonon']:
        q_pos = p['q-position']
        q_pts.append(q_pos)
        m = p.get('multiplicity', 1)
        multiplicities.append(m)
        bands, gruneisenband = ([], [])
        for b in p['band']:
            bands.append(b['frequency'])
            if 'gruneisen' in b:
                gruneisenband.append(b['gruneisen'])
        frequencies.append(bands)
        gruneisen.append(gruneisenband)
        if 'label' in p:
            phonopy_labels_dict[p['label']] = p['q-position']
    q_pts_np = np.array(q_pts)
    multiplicities_np = np.array(multiplicities)
    frequencies_np = np.transpose(frequencies)
    gruneisen_np = np.transpose(gruneisen)
    return GruneisenParameter(gruneisen=gruneisen_np, qpoints=q_pts_np, multiplicities=multiplicities_np, frequencies=frequencies_np, structure=structure)