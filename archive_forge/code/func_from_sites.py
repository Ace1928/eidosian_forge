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
@classmethod
def from_sites(cls, sites: Sequence[Site], charge: float=0, spin_multiplicity: int | None=None, validate_proximity: bool=False, charge_spin_check: bool=True, properties: dict | None=None) -> IMolecule | Molecule:
    """Convenience constructor to make a Molecule from a list of sites.

        Args:
            sites ([Site]): Sequence of Sites.
            charge (int): Charge of molecule. Defaults to 0.
            spin_multiplicity (int): Spin multicipity. Defaults to None,
                in which it is determined automatically.
            validate_proximity (bool): Whether to check that atoms are too
                close.
            charge_spin_check (bool): Whether to check that the charge and
                spin multiplicity are compatible with each other. Defaults
                to True.
            properties (dict): dictionary containing properties associated
                with the whole molecule.

        Raises:
            ValueError: If sites is empty

        Returns:
            Molecule
        """
    if len(sites) < 1:
        raise ValueError(f'You need at least 1 site to make a {cls.__name__}')
    props = collections.defaultdict(list)
    for site in sites:
        for k, v in site.properties.items():
            props[k].append(v)
    labels = [site.label for site in sites]
    return cls([site.species for site in sites], [site.coords for site in sites], charge=charge, spin_multiplicity=spin_multiplicity, validate_proximity=validate_proximity, site_properties=props, labels=labels, charge_spin_check=charge_spin_check, properties=properties)