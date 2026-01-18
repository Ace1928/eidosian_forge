from __future__ import annotations
import abc
from collections import namedtuple
from collections.abc import Iterable
from enum import Enum, unique
from pprint import pformat
from typing import TYPE_CHECKING, cast
import numpy as np
from monty.collections import AttrDict
from monty.design_patterns import singleton
from monty.json import MontyDecoder, MontyEncoder, MSONable
from pymatgen.core import ArrayWithUnit, Lattice, Species, Structure, units
def structure_to_abivars(structure: Structure, enforce_znucl: list | None=None, enforce_typat: list | None=None, **kwargs):
    """
    Receives a structure and returns a dictionary with ABINIT variables.

    Args:
        enforce_znucl (list): ntypat entries with the value of Z for each type of atom.
            Used to change the default ordering. Defaults to None.
        enforce_typat (list): natom entries with the type index.
            Fortran conventions: start to count from 1. Used to change the default ordering.
    """
    if not structure.is_ordered:
        raise ValueError('Received disordered structure with partial occupancies that cannot be converted into an Abinit input. Please use OrderDisorderedStructureTransformation or EnumerateStructureTransformation to build an appropriate supercell from partial occupancies or, alternatively, use the Rigid Band Model or the Virtual Crystal Approximation.')
    n_atoms = len(structure)
    enforce_order = False
    if enforce_znucl is not None or enforce_typat is not None:
        enforce_order = True
        if enforce_znucl is None or enforce_typat is None:
            raise ValueError('Both enforce_znucl and enforce_typat are required!')
        if len(enforce_typat) != len(structure):
            raise ValueError(f'enforce_typat contains {len(enforce_typat)} entries while it should be len(structure)={len(structure)!r}')
        if len(enforce_znucl) != structure.n_elems:
            raise ValueError(f'enforce_znucl contains {len(enforce_znucl)} entries while it should be structure.n_elems={structure.n_elems!r}')
    if enforce_order:
        znucl_type = enforce_znucl
        typat = enforce_typat or []
    else:
        types_of_specie = species_by_znucl(structure)
        znucl_type = [specie.number for specie in types_of_specie]
        typat = np.zeros(n_atoms, int)
        for atm_idx, site in enumerate(structure):
            typat[atm_idx] = types_of_specie.index(site.specie) + 1
    r_prim = ArrayWithUnit(structure.lattice.matrix, 'ang').to('bohr')
    ang_deg = structure.lattice.angles
    x_red = np.reshape([site.frac_coords for site in structure], (-1, 3))
    r_prim = np.where(np.abs(r_prim) > 1e-08, r_prim, 0.0)
    x_red = np.where(np.abs(x_red) > 1e-08, x_red, 0.0)
    dct = {'natom': n_atoms, 'ntypat': structure.n_elems, 'typat': typat, 'znucl': znucl_type, 'xred': x_red}
    geo_mode = kwargs.pop('geomode', 'rprim')
    if geo_mode == 'automatic':
        geo_mode = 'rprim'
        if structure.lattice.is_hexagonal():
            geo_mode = 'angdeg'
            ang_deg = structure.lattice.angles
    if geo_mode == 'rprim':
        dct.update(acell=3 * [1.0], rprim=r_prim)
    elif geo_mode == 'angdeg':
        dct.update(acell=ArrayWithUnit(structure.lattice.abc, 'ang').to('bohr'), angdeg=ang_deg)
    else:
        raise ValueError(f'Wrong value for geo_mode={geo_mode!r}')
    return dct