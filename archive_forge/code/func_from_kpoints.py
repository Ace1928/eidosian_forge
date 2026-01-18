from __future__ import annotations
import abc
import copy
import hashlib
import itertools
import os
import re
import textwrap
import typing
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from monty.dev import deprecated
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core import Element
from pymatgen.io.cp2k.utils import chunk, postprocessor, preprocessor
from pymatgen.io.vasp.inputs import Kpoints as VaspKpoints
from pymatgen.io.vasp.inputs import KpointsSupportedModes
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
@classmethod
def from_kpoints(cls, kpoints: VaspKpoints, kpoints_line_density: int=20) -> Self:
    """
        Initialize band structure section from a line-mode Kpoint object.

        Args:
            kpoints: a kpoint object from the vasp module, which was constructed in line mode
            kpoints_line_density: Number of kpoints along each path
        """
    if kpoints.style == KpointsSupportedModes.Line_mode:

        def pairwise(iterable):
            a = iter(iterable)
            return zip(a, a)
        kpoint_sets = [KpointSet(npoints=kpoints_line_density, kpoints=[(lbls[0], kpts[0]), (lbls[1], kpts[1])], units='B_VECTOR') for lbls, kpts in zip(pairwise(kpoints.labels), pairwise(kpoints.kpts))]
    elif kpoints.style in (KpointsSupportedModes.Reciprocal, KpointsSupportedModes.Cartesian):
        kpoint_sets = [KpointSet(npoints=1, kpoints=[('None', kpts) for kpts in kpoints.kpts], units='B_VECTOR' if kpoints.coord_type == 'Reciprocal' else 'CART_ANGSTROM')]
    else:
        raise ValueError('Unsupported k-point style. Must be line-mode or explicit k-points (reciprocal/cartesian).')
    return cls(kpoint_sets=kpoint_sets, filename='BAND.bs')