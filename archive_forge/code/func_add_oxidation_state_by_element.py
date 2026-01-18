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
def add_oxidation_state_by_element(self, oxidation_states: dict[str, float]) -> SiteCollection:
    """Add oxidation states.

        Args:
            oxidation_states (dict): Dict of oxidation states.
                E.g., {"Li":1, "Fe":2, "P":5, "O":-2}

        Raises:
            ValueError if oxidation states are not specified for all elements.

        Returns:
            SiteCollection: self with oxidation states.
        """
    if (missing := ({el.symbol for el in self.composition} - {*oxidation_states})):
        raise ValueError(f'Oxidation states not specified for all elements, missing={missing!r}')
    for site in self:
        new_sp = {}
        for el, occu in site.species.items():
            new_sp[Species(el.symbol, oxidation_states[el.symbol])] = occu
        site.species = Composition(new_sp)
    return self