from __future__ import annotations
import logging
import os
import warnings
from collections import namedtuple
from enum import Enum, unique
from typing import TYPE_CHECKING, Any, no_type_check
import numpy as np
from monty.serialization import loadfn
from ruamel.yaml.error import MarkedYAMLError
from scipy.signal import argrelextrema
from scipy.stats import gaussian_kde
from pymatgen.core.structure import DummySpecies, Element, Species, Structure
from pymatgen.electronic_structure.core import Magmom
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.transformations.advanced_transformations import MagOrderingTransformation, MagOrderParameterConstraint
from pymatgen.transformations.standard_transformations import AutoOxiStateDecorationTransformation
from pymatgen.util.due import Doi, due
@due.dcite(Doi('10.1021/acs.chemmater.6b04729'), description='A Simple Computational Proxy for Screening Magnetocaloric Compounds')
def magnetic_deformation(structure_A: Structure, structure_B: Structure) -> MagneticDeformation:
    """Calculates 'magnetic deformation proxy',
    a measure of deformation (norm of finite strain)
    between 'non-magnetic' (non-spin-polarized) and
    ferromagnetic structures.

    Adapted from Bocarsly et al. 2017,
    doi: 10.1021/acs.chemmater.6b04729

    Args:
        structure_A: Structure
        structure_B: Structure

    Returns:
        MagneticDeformation
    """
    ordering_a = CollinearMagneticStructureAnalyzer(structure_A, overwrite_magmom_mode='none').ordering
    ordering_b = CollinearMagneticStructureAnalyzer(structure_B, overwrite_magmom_mode='none').ordering
    type_str = f'{ordering_a.value}-{ordering_b.value}'
    lattice_a = structure_A.lattice.matrix.T
    lattice_b = structure_B.lattice.matrix.T
    lattice_a_inv = np.linalg.inv(lattice_a)
    p = np.dot(lattice_a_inv, lattice_b)
    eta = 0.5 * (np.dot(p.T, p) - np.identity(3))
    w, _v = np.linalg.eig(eta)
    deformation = 100 * (1 / 3) * np.sqrt(w[0] ** 2 + w[1] ** 2 + w[2] ** 2)
    return MagneticDeformation(deformation=deformation, type=type_str)