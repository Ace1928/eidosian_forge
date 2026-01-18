from __future__ import annotations
import json
import math
import os
import warnings
from bisect import bisect_left
from collections import defaultdict, namedtuple
from copy import deepcopy
from functools import lru_cache
from math import acos, asin, atan2, cos, exp, fabs, pi, pow, sin, sqrt
from typing import TYPE_CHECKING, Any, Literal, get_args
import numpy as np
from monty.dev import deprecated, requires
from monty.serialization import loadfn
from ruamel.yaml import YAML
from scipy.spatial import Voronoi
from pymatgen.analysis.bond_valence import BV_PARAMS, BVAnalyzer
from pymatgen.analysis.graphs import MoleculeGraph, StructureGraph
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
from pymatgen.core import Element, IStructure, PeriodicNeighbor, PeriodicSite, Site, Species, Structure
@classmethod
def from_preset(cls, preset) -> Self:
    """
        Initialize a CutOffDictNN according to a preset set of cutoffs.

        Args:
            preset (str): A preset name. The list of supported presets are:
                - "vesta_2019": The distance cutoffs used by the VESTA visualisation program.

        Returns:
            A CutOffDictNN using the preset cut-off dictionary.
        """
    if preset == 'vesta_2019':
        cut_offs = loadfn(f'{module_dir}/vesta_cutoffs.yaml')
        return cls(cut_off_dict=cut_offs)
    raise ValueError(f'Unknown preset={preset!r}')