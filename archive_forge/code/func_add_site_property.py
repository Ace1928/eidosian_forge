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
def add_site_property(self, property_name: str, values: Sequence | np.ndarray) -> SiteCollection:
    """Adds a property to a site. Note: This is the preferred method
        for adding magnetic moments, selective dynamics, and related
        site-specific properties to a structure/molecule object.

        Examples:
            structure.add_site_property("magmom", [1.0, 0.0])
            structure.add_site_property("selective_dynamics", [[True, True, True], [False, False, False]])

        Args:
            property_name (str): The name of the property to add.
            values (list): A sequence of values. Must be same length as
                number of sites.

        Raises:
            ValueError: if len(values) != number of sites.

        Returns:
            SiteCollection: self with site property added.
        """
    if len(values) != len(self):
        raise ValueError(f'len(values)={len(values)!r} must equal sites in structure={len(self)}')
    for site, val in zip(self, values):
        site.properties[property_name] = val
    return self