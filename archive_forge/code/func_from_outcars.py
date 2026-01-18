from __future__ import annotations
import os
from glob import glob
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from monty.json import MSONable, jsanitize
from scipy.interpolate import CubicSpline
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.io.vasp import Outcar
from pymatgen.util.plotting import pretty_plot
@classmethod
def from_outcars(cls, outcars, structures, **kwargs) -> Self:
    """
        Initializes an NEBAnalysis from Outcar and Structure objects. Use
        the static constructors, e.g., from_dir instead if you
        prefer to have these automatically generated from a directory of NEB
        calculations.

        Args:
            outcars ([Outcar]): List of Outcar objects. Note that these have
                to be ordered from start to end along reaction coordinates.
            structures ([Structure]): List of Structures along reaction
                coordinate. Must be same length as outcar.
            interpolation_order (int): Order of polynomial to use to
                interpolate between images. Same format as order parameter in
                scipy.interplotate.PiecewisePolynomial.
        """
    if len(outcars) != len(structures):
        raise ValueError('# of Outcars must be same as # of Structures')
    rms_dist = [0]
    prev = structures[0]
    for st in structures[1:]:
        dists = np.array([s2.distance(s1) for s1, s2 in zip(prev, st)])
        rms_dist.append(np.sqrt(np.sum(dists ** 2)))
        prev = st
    rms_dist = np.cumsum(rms_dist)
    energies = []
    forces = []
    for idx, outcar in enumerate(outcars):
        outcar.read_neb()
        energies.append(outcar.data['energy'])
        if idx in [0, len(outcars) - 1]:
            forces.append(0)
        else:
            forces.append(outcar.data['tangent_force'])
    forces = np.array(forces)
    rms_dist = np.array(rms_dist)
    return cls(r=rms_dist, energies=energies, forces=forces, structures=structures, **kwargs)