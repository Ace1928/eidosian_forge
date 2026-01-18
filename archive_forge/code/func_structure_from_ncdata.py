from __future__ import annotations
import logging
import os.path
import warnings
from typing import TYPE_CHECKING
import numpy as np
from monty.collections import AttrDict
from monty.dev import requires
from monty.functools import lazy_property
from monty.string import marquee
from pymatgen.core.structure import Structure
from pymatgen.core.units import ArrayWithUnit
from pymatgen.core.xcfunc import XcFunc
def structure_from_ncdata(ncdata, site_properties=None, cls=Structure):
    """
    Reads and returns a pymatgen structure from a NetCDF file
    containing crystallographic data in the ETSF-IO format.

    Args:
        ncdata: filename or NetcdfReader instance.
        site_properties: Dictionary with site properties.
        cls: The Structure class to instantiate.
    """
    ncdata, close_it = as_ncreader(ncdata)
    lattice = ArrayWithUnit(ncdata.read_value('primitive_vectors'), 'bohr').to('ang')
    red_coords = ncdata.read_value('reduced_atom_positions')
    n_atom = len(red_coords)
    znucl_type = ncdata.read_value('atomic_numbers')
    type_atom = ncdata.read_value('atom_species')
    species = n_atom * [None]
    for atom in range(n_atom):
        type_idx = type_atom[atom] - 1
        species[atom] = int(znucl_type[type_idx])
    dct = {}
    if site_properties is not None:
        for prop in site_properties:
            dct[prop] = ncdata.read_value(prop)
    structure = cls(lattice, species, red_coords, site_properties=dct)
    try:
        from abipy.core.structure import Structure as AbipyStructure
        structure.__class__ = AbipyStructure
    except ImportError:
        pass
    if close_it:
        ncdata.close()
    return structure