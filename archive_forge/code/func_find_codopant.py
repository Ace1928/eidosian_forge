from __future__ import annotations
import logging
import math
import warnings
from fractions import Fraction
from itertools import groupby, product
from math import gcd
from string import ascii_lowercase
from typing import TYPE_CHECKING, Callable, Literal
import numpy as np
from joblib import Parallel, delayed
from monty.dev import requires
from monty.fractions import lcm
from monty.json import MSONable
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.energy_models import SymmetryModel
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.analysis.gb.grain import GrainBoundaryGenerator
from pymatgen.analysis.local_env import MinimumDistanceNN
from pymatgen.analysis.structure_matcher import SpinComparator, StructureMatcher
from pymatgen.analysis.structure_prediction.substitution_probability import SubstitutionPredictor
from pymatgen.command_line.enumlib_caller import EnumError, EnumlibAdaptor
from pymatgen.command_line.mcsqs_caller import run_mcsqs
from pymatgen.core import DummySpecies, Element, Species, Structure, get_el_sp
from pymatgen.core.surface import SlabGenerator
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.icet import IcetSQS
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.standard_transformations import (
from pymatgen.transformations.transformation_abc import AbstractTransformation
def find_codopant(target: Species, oxidation_state: float, allowed_elements: Sequence[str] | None=None) -> Species:
    """Finds the element from "allowed elements" that (i) possesses the desired
    "oxidation state" and (ii) is closest in ionic radius to the target specie.

    Args:
        target (Species): provides target ionic radius.
        oxidation_state (float): co-dopant oxidation state.
        allowed_elements (list[str]): List of allowed elements. If None,
            all elements are tried.

    Returns:
        Species: with oxidation_state that has ionic radius closest to target.
    """
    ref_radius = target.ionic_radius
    if ref_radius is None:
        raise ValueError(f'Target species {target} has no ionic radius.')
    candidates: list[tuple[float, Species]] = []
    symbols = allowed_elements or [el.symbol for el in Element]
    for sym in symbols:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                sp = Species(sym, oxidation_state)
                radius = sp.ionic_radius
                if radius is not None:
                    candidates.append((radius, sp))
        except Exception:
            pass
    return min(candidates, key=lambda tup: abs(tup[0] / ref_radius - 1))[1]