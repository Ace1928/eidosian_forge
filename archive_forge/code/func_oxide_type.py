from __future__ import annotations
import collections
import itertools
from math import acos, pi
from typing import TYPE_CHECKING
from warnings import warn
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi
from pymatgen.analysis.local_env import JmolNN, VoronoiNN
from pymatgen.core import Composition, Element, PeriodicSite, Species
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def oxide_type(structure: Structure, relative_cutoff: float=1.1, return_nbonds: bool=False) -> str | tuple[str, int]:
    """
    Determines if an oxide is a peroxide/superoxide/ozonide/normal oxide.

    Args:
        structure (Structure): Input structure.
        relative_cutoff (float): Relative_cutoff * act. cutoff stipulates the
            max distance two O atoms must be from each other.
        return_nbonds (bool): Should number of bonds be requested?
    """
    ox_obj = OxideType(structure, relative_cutoff)
    if return_nbonds:
        return (ox_obj.oxide_type, ox_obj.nbonds)
    return ox_obj.oxide_type