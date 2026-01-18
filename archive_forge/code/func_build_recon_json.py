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
def build_recon_json() -> dict:
    """Build reconstruction instructions, optionally upon a base instruction set."""
    if reconstruction_name not in RECONSTRUCTIONS_ARCHIVE:
        raise KeyError(f"reconstruction_name={reconstruction_name!r} does not exist in the archive. Please select from one of the following: {list(RECONSTRUCTIONS_ARCHIVE)} or add it to the archive file 'reconstructions_archive.json'.")
    recon_json: dict = copy.deepcopy(RECONSTRUCTIONS_ARCHIVE[reconstruction_name])
    if 'base_reconstruction' in recon_json:
        new_points_to_add: list = []
        new_points_to_remove: list = []
        if 'points_to_add' in recon_json:
            new_points_to_add = recon_json['points_to_add']
        if 'points_to_remove' in recon_json:
            new_points_to_remove = recon_json['points_to_remove']
        recon_json = copy.deepcopy(RECONSTRUCTIONS_ARCHIVE[recon_json['base_reconstruction']])
        if 'points_to_add' in recon_json:
            del recon_json['points_to_add']
        if new_points_to_add:
            recon_json['points_to_add'] = new_points_to_add
        if 'points_to_remove' in recon_json:
            del recon_json['points_to_remove']
        if new_points_to_remove:
            recon_json['points_to_remove'] = new_points_to_remove
    return recon_json