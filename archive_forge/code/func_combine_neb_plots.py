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
def combine_neb_plots(neb_analyses, arranged_neb_analyses=False, reverse_plot=False):
    """
    neb_analyses: a list of NEBAnalysis objects.

    arranged_neb_analyses: The code connects two end points with the
    smallest-energy difference. If all end points have very close energies, it's
    likely to result in an inaccurate connection. Manually arrange neb_analyses
    if the combined plot is not as expected compared with all individual plots.
    E.g., if there are two NEBAnalysis objects to combine, arrange in such a
    way that the end-point energy of the first NEBAnalysis object is the
    start-point energy of the second NEBAnalysis object.
    Note that the barrier labeled in y-axis in the combined plot might be
    different from that in the individual plot due to the reference energy used.
    reverse_plot: reverse the plot or percolation direction.

    Returns:
        a NEBAnalysis object
    """
    x = StructureMatcher()
    for neb_index, neb in enumerate(neb_analyses):
        if neb_index == 0:
            neb1 = neb
            neb1_energies = list(neb1.energies)
            neb1_structures = neb1.structures
            neb1_forces = neb1.forces
            neb1_r = neb1.r
            continue
        neb2 = neb
        neb2_energies = list(neb2.energies)
        matching = 0
        for neb1_s in [neb1_structures[0], neb1_structures[-1]]:
            if x.fit(neb1_s, neb2.structures[0]) or x.fit(neb1_s, neb2.structures[-1]):
                matching += 1
                break
        if matching == 0:
            raise ValueError('no matched structures for connection!')
        neb1_start_e, neb1_end_e = (neb1_energies[0], neb1_energies[-1])
        neb2_start_e, neb2_end_e = (neb2_energies[0], neb2_energies[-1])
        min_e_diff = min([abs(neb1_start_e - neb2_start_e), abs(neb1_start_e - neb2_end_e), abs(neb1_end_e - neb2_start_e), abs(neb1_end_e - neb2_end_e)])
        if arranged_neb_analyses:
            neb1_energies = neb1_energies[0:len(neb1_energies) - 1] + [(neb1_energies[-1] + neb2_energies[0]) / 2] + neb2_energies[1:]
            neb1_structures = neb1_structures + neb2.structures[1:]
            neb1_forces = list(neb1_forces) + list(neb2.forces)[1:]
            neb1_r = list(neb1_r) + [i + neb1_r[-1] for i in list(neb2.r)[1:]]
        elif abs(neb1_start_e - neb2_start_e) == min_e_diff:
            neb1_energies = list(reversed(neb1_energies[1:])) + neb2_energies
            neb1_structures = list(reversed(neb1_structures[1:])) + neb2.structures
            neb1_forces = list(reversed(list(neb1_forces)[1:])) + list(neb2.forces)
            neb1_r = list(reversed([i * -1 - neb1_r[-1] * -1 for i in list(neb1_r)[1:]])) + [i + neb1_r[-1] for i in list(neb2.r)]
        elif abs(neb1_start_e - neb2_end_e) == min_e_diff:
            neb1_energies = neb2_energies + neb1_energies[1:]
            neb1_structures = neb2.structures + neb1_structures[1:]
            neb1_forces = list(neb2.forces) + list(neb1_forces)[1:]
            neb1_r = list(neb2.r) + [i + list(neb2.r)[-1] for i in list(neb1_r)[1:]]
        elif abs(neb1_end_e - neb2_start_e) == min_e_diff:
            neb1_energies = neb1_energies + neb2_energies[1:]
            neb1_structures = neb1_structures + neb2.structures[1:]
            neb1_forces = list(neb1_forces) + list(neb2.forces)[1:]
            neb1_r = list(neb1_r) + [i + neb1_r[-1] for i in list(neb2.r)[1:]]
        else:
            neb1_energies = neb1_energies + list(reversed(neb2_energies))[1:]
            neb1_structures = neb1_structures + list(reversed(neb2.structures))[1:]
            neb1_forces = list(neb1_forces) + list(reversed(list(neb2.forces)))[1:]
            neb1_r = list(neb1_r) + list(reversed([i * -1 - list(neb2.r)[-1] * -1 + list(neb1_r)[-1] for i in list(neb2.r)[:-1]]))
    if reverse_plot:
        na = NEBAnalysis(list(reversed([i * -1 - neb1_r[-1] * -1 for i in list(neb1_r)])), list(reversed(neb1_energies)), list(reversed(neb1_forces)), list(reversed(neb1_structures)))
    else:
        na = NEBAnalysis(neb1_r, neb1_energies, neb1_forces, neb1_structures)
    return na