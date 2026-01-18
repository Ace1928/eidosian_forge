import os
import numpy as np
from ase.units import Bohr, Ha, Ry, fs, m, s
from ase.calculators.calculator import kpts2sizeandoffsets
from ase.calculators.openmx.reader import (read_electron_valency, get_file_name, get_standard_key)
from ase.calculators.openmx import parameters as param
def get_definition_of_atomic_species(atoms, parameters):
    """
    Using atoms and parameters, Returns the list `definition_of_atomic_species`
    where matrix of strings contains the information between keywords.
    For example,
     definition_of_atomic_species =
         [['H','H5.0-s1>1p1>1','H_CA13'],
          ['C','C5.0-s1>1p1>1','C_CA13']]
    Goes to,
      <Definition.of.Atomic.Species
        H   H5.0-s1>1p1>1      H_CA13
        C   C5.0-s1>1p1>1      C_CA13
      Definition.of.Atomic.Species>
    Further more, you can specify the wannier information here.
    A. Define local functions for projectors
      Since the pseudo-atomic orbitals are used for projectors,
      the specification of them is the same as for the basis functions.
      An example setting, for silicon in diamond structure, is as following:
   Species.Number          2
      <Definition.of.Atomic.Species
        Si       Si7.0-s2p2d1    Si_CA13
        proj1    Si5.5-s1p1d1f1  Si_CA13
      Definition.of.Atomic.Species>
    """
    if parameters.get('definition_of_atomic_species') is not None:
        return parameters['definition_of_atomic_species']
    definition_of_atomic_species = []
    xc = parameters.get('_xc')
    year = parameters.get('_year')
    chem = atoms.get_chemical_symbols()
    species = get_species(chem)
    for element in species:
        rad_orb = get_cutoff_radius_and_orbital(element=element)
        suffix = get_pseudo_potential_suffix(element=element, xc=xc, year=year)
        definition_of_atomic_species.append([element, rad_orb, suffix])
    wannier_projectors = parameters.get('definition_of_wannier_projectors', [])
    for i, projector in enumerate(wannier_projectors):
        full_projector = definition_of_atomic_species[i]
        full_projector[0] = projector
        definition_of_atomic_species.append(full_projector)
    return definition_of_atomic_species