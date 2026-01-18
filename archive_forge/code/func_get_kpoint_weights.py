from __future__ import annotations
import copy
import itertools
import logging
import math
import warnings
from collections import defaultdict
from collections.abc import Sequence
from fractions import Fraction
from functools import lru_cache
from math import cos, sin
from typing import TYPE_CHECKING, Any, Literal
import numpy as np
import scipy.cluster
import spglib
from pymatgen.core.lattice import Lattice
from pymatgen.core.operations import SymmOp
from pymatgen.core.structure import Molecule, PeriodicSite, Structure
from pymatgen.symmetry.structure import SymmetrizedStructure
from pymatgen.util.coord import find_in_coord_list, pbc_diff
from pymatgen.util.due import Doi, due
def get_kpoint_weights(self, kpoints, atol=1e-05):
    """Calculate the weights for a list of kpoints.

        Args:
            kpoints (Sequence): Sequence of kpoints. np.arrays is fine. Note
                that the code does not check that the list of kpoints
                provided does not contain duplicates.
            atol (float): Tolerance for fractional coordinates comparisons.

        Returns:
            List of weights, in the SAME order as kpoints.
        """
    kpts = np.array(kpoints)
    shift = []
    mesh = []
    for idx in range(3):
        nonzero = [i for i in kpts[:, idx] if abs(i) > 1e-05]
        if len(nonzero) != len(kpts):
            if not nonzero:
                mesh.append(1)
            else:
                m = np.abs(np.round(1 / np.array(nonzero)))
                mesh.append(int(max(m)))
            shift.append(0)
        else:
            m = np.abs(np.round(0.5 / np.array(nonzero)))
            mesh.append(int(max(m)))
            shift.append(1)
    mapping, grid = spglib.get_ir_reciprocal_mesh(np.array(mesh), self._cell, is_shift=shift, symprec=self._symprec)
    mapping = list(mapping)
    grid = (np.array(grid) + np.array(shift) * (0.5, 0.5, 0.5)) / mesh
    weights = []
    mapped = defaultdict(int)
    for kpt in kpoints:
        for idx, g in enumerate(grid):
            if np.allclose(pbc_diff(kpt, g), (0, 0, 0), atol=atol):
                mapped[tuple(g)] += 1
                weights.append(mapping.count(mapping[idx]))
                break
    if len(mapped) != len(set(mapping)) or not all((v == 1 for v in mapped.values())):
        raise ValueError('Unable to find 1:1 corresponding between input kpoints and irreducible grid!')
    return [w / sum(weights) for w in weights]