from __future__ import annotations
import copy
import itertools
import json
import logging
import math
import os
import warnings
from functools import reduce
from math import gcd, isclose
from typing import TYPE_CHECKING, cast
import numpy as np
from monty.fractions import lcm
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, PeriodicSite, Structure, get_el_sp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.coord import in_coord_list
from pymatgen.util.due import Doi, due
def generate_all_slabs(structure: Structure, max_index: int, min_slab_size: float, min_vacuum_size: float, bonds: dict | None=None, tol: float=0.1, ftol: float=0.1, max_broken_bonds: int=0, lll_reduce: bool=False, center_slab: bool=False, primitive: bool=True, max_normal_search: int | None=None, symmetrize: bool=False, repair: bool=False, include_reconstructions: bool=False, in_unit_planes: bool=False) -> list[Slab]:
    """Find all unique Slabs up to a given Miller index.

    Slabs oriented along certain Miller indices may be equivalent to
    other Miller indices under symmetry operations. To avoid
    duplication, such equivalent slabs would be filtered out.
    For instance, CsCl has equivalent slabs in the (0,0,1),
    (0,1,0), and (1,0,0) directions under symmetry operations.

    Args:
        structure (Structure): Initial input structure. To
            ensure that the Miller indices correspond to usual
            crystallographic definitions, you should supply a
            conventional unit cell.
        max_index (int): The maximum Miller index to go up to.
        min_slab_size (float): The minimum slab size in Angstrom.
        min_vacuum_size (float): The minimum vacuum layer thickness in Angstrom.
        bonds (dict): A {(species1, species2): max_bond_dist} dict.
                For example, PO4 groups may be defined as {("P", "O"): 3}.
        tol (float): Tolerance for getting primitive cells and
            matching structures.
        ftol (float): Tolerance in Angstrom for fcluster to check
            if two atoms are on the same plane. Default to 0.1 Angstrom
            in the direction of the surface normal.
        max_broken_bonds (int): Maximum number of allowable broken bonds
            for the slab. Use this to limit the number of slabs.
            Defaults to zero, which means no bond can be broken.
        lll_reduce (bool): Whether to perform an LLL reduction on the
            final Slab.
        center_slab (bool): Whether to center the slab in the cell with
            equal vacuum spacing from the top and bottom.
        primitive (bool): Whether to reduce generated slabs to
            primitive cell. Note this does NOT generate a slab
            from a primitive cell, it means that after slab
            generation, we attempt to reduce the generated slab to
            primitive cell.
        max_normal_search (int): If set to a positive integer, the code
            will search for a normal lattice vector that is as
            perpendicular to the surface as possible, by considering
            multiple linear combinations of lattice vectors up to
            this value. This has no bearing on surface energies,
            but may be useful as a preliminary step to generate slabs
            for absorption or other sizes. It may not be the smallest possible
            cell for simulation. Normality is not guaranteed, but the oriented
            cell will have the c vector as normal as possible to the surface.
            The max absolute Miller index is usually sufficient.
        symmetrize (bool): Whether to ensure the surfaces of the
            slabs are equivalent.
        repair (bool): Whether to repair terminations with broken bonds
            or just omit them.
        include_reconstructions (bool): Whether to include reconstructed
            slabs available in the reconstructions_archive.json file. Defaults to False.
        in_unit_planes (bool): Whether to set min_slab_size and min_vac_size
            in number of hkl planes or Angstrom (default).
            Setting in units of planes is useful to ensure some slabs
            to have a certain number of layers, e.g. for Cs(100), 10 Ang
            will result in a slab with only 2 layers, whereas
            Fe(100) will have more layers. The slab thickness
            will be in min_slab_size/math.ceil(self._proj_height/dhkl)
            multiples of oriented unit cells.
    """
    all_slabs = []
    for miller in get_symmetrically_distinct_miller_indices(structure, max_index):
        gen = SlabGenerator(structure, miller, min_slab_size, min_vacuum_size, lll_reduce=lll_reduce, center_slab=center_slab, primitive=primitive, max_normal_search=max_normal_search, in_unit_planes=in_unit_planes)
        slabs = gen.get_slabs(bonds=bonds, tol=tol, ftol=ftol, symmetrize=symmetrize, max_broken_bonds=max_broken_bonds, repair=repair)
        if len(slabs) > 0:
            logger.debug(f'{miller} has {len(slabs)} slabs... ')
            all_slabs.extend(slabs)
    if include_reconstructions:
        symbol = SpacegroupAnalyzer(structure).get_space_group_symbol()
        for name, instructions in RECONSTRUCTIONS_ARCHIVE.items():
            if 'base_reconstruction' in instructions:
                instructions = RECONSTRUCTIONS_ARCHIVE[instructions['base_reconstruction']]
            if instructions['spacegroup']['symbol'] == symbol:
                if max(instructions['miller_index']) > max_index:
                    continue
                recon = ReconstructionGenerator(structure, min_slab_size, min_vacuum_size, name)
                all_slabs.extend(recon.build_slabs())
    return all_slabs