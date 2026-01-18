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
def set_charge_and_spin(self, charge: float, spin_multiplicity: int | None=None) -> Molecule:
    """Set the charge and spin multiplicity.

        Args:
            charge (int): Charge for the molecule. Defaults to 0.
            spin_multiplicity (int): Spin multiplicity for molecule.
                Defaults to None, which means that the spin multiplicity is
                set to 1 if the molecule has no unpaired electrons and to 2
                if there are unpaired electrons.

        Returns:
            Molecule: self with new charge and spin multiplicity set.
        """
    self._charge = charge
    n_electrons = 0.0
    for site in self:
        for sp, amt in site.species.items():
            if not isinstance(sp, DummySpecies):
                n_electrons += sp.Z * amt
    n_electrons -= charge
    self._nelectrons = n_electrons
    if spin_multiplicity:
        if self._charge_spin_check and (n_electrons + spin_multiplicity) % 2 != 1:
            raise ValueError(f'Charge of {self._charge} and spin multiplicity of {spin_multiplicity} is not possible for this molecule')
        self._spin_multiplicity = spin_multiplicity
    else:
        self._spin_multiplicity = 1 if n_electrons % 2 == 0 else 2
    return self