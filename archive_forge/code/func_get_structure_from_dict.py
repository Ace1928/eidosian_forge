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
def get_structure_from_dict(dct):
    """
    Extracts a structure from the dictionary extracted from the output
    files of phonopy like phonopy.yaml or band.yaml.
    Adds "phonopy_masses" in the site_properties of the structures.
    Compatible with older phonopy versions.
    """
    species = []
    frac_coords = []
    masses = []
    if 'points' in dct:
        for pt in dct['points']:
            species.append(pt['symbol'])
            frac_coords.append(pt['coordinates'])
            masses.append(pt['mass'])
    elif 'atoms' in dct:
        for pt in dct['atoms']:
            species.append(pt['symbol'])
            frac_coords.append(pt['position'])
            masses.append(pt['mass'])
    else:
        raise ValueError('The dict does not contain structural information')
    return Structure(dct['lattice'], species, frac_coords, site_properties={'phonopy_masses': masses})