from __future__ import annotations
import collections
import contextlib
import functools
import inspect
import io
import itertools
import json
import math
import os
import random
import re
import sys
import warnings
from abc import ABC, abstractmethod
from fnmatch import fnmatch
from inspect import isclass
from io import StringIO
from typing import TYPE_CHECKING, Any, Callable, Literal, SupportsIndex, cast, get_args
import numpy as np
from monty.dev import deprecated
from monty.io import zopen
from monty.json import MSONable
from numpy import cross, eye
from numpy.linalg import norm
from ruamel.yaml import YAML
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.linalg import expm, polar
from scipy.spatial.distance import squareform
from tabulate import tabulate
from pymatgen.core.bonds import CovalentBond, get_bond_length
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice, get_points_in_spheres
from pymatgen.core.operations import SymmOp
from pymatgen.core.periodic_table import DummySpecies, Element, Species, get_el_sp
from pymatgen.core.sites import PeriodicSite, Site
from pymatgen.core.units import Length, Mass
from pymatgen.electronic_structure.core import Magmom
from pymatgen.symmetry.maggroups import MagneticSpaceGroup
from pymatgen.util.coord import all_distances, get_angle, lattice_points_in_supercell
def get_space_group_info(self, symprec: float=0.01, angle_tolerance: float=5.0) -> tuple[str, int]:
    """Convenience method to quickly get the spacegroup of a structure.

        Args:
            symprec (float): Same definition as in SpacegroupAnalyzer.
                Defaults to 1e-2.
            angle_tolerance (float): Same definition as in SpacegroupAnalyzer.
                Defaults to 5 degrees.

        Returns:
            spacegroup_symbol, international_number
        """
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    spg_analyzer = SpacegroupAnalyzer(self, symprec=symprec, angle_tolerance=angle_tolerance)
    return (spg_analyzer.get_space_group_symbol(), spg_analyzer.get_space_group_number())