from __future__ import annotations
import copy
import itertools
import random
import warnings
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from sympy import Symbol
from sympy.solvers import linsolve, solve
from pymatgen.analysis.wulff import WulffShape
from pymatgen.core import Structure
from pymatgen.core.composition import Composition
from pymatgen.core.surface import get_slab_regions
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.vasp.outputs import Locpot, Outcar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.due import Doi, due
from pymatgen.util.plotting import pretty_plot
def surface_chempot_range_map(self, elements, miller_index, ranges, incr=50, no_doped=False, no_clean=False, delu_dict=None, ax=None, annotate=True, show_unphysical_only=False, fontsize=10) -> plt.Axes:
    """
        Adapted from the get_chempot_range_map() method in the PhaseDiagram
            class. Plot the chemical potential range map based on surface
            energy stability. Currently works only for 2-component PDs. At
            the moment uses a brute force method by enumerating through the
            range of the first element chempot with a specified increment
            and determines the chempot range of the second element for each
            SlabEntry. Future implementation will determine the chempot range
            map first by solving systems of equations up to 3 instead of 2.

        Args:
            elements (list): Sequence of elements to be considered as independent
                variables. E.g., if you want to show the stability ranges of
                all Li-Co-O phases w.r.t. to duLi and duO, you will supply
                [Element("Li"), Element("O")]
            miller_index ([h, k, l]): Miller index of the surface we are interested in
            ranges ([[range1], [range2]]): List of chempot ranges (max and min values)
                for the first and second element.
            incr (int): Number of points to sample along the range of the first chempot
            no_doped (bool): Whether or not to include doped systems.
            no_clean (bool): Whether or not to include clean systems.
            delu_dict (dict): Dictionary of the chemical potentials to be set as
                constant. Note the key should be a sympy Symbol object of the
                format: Symbol("delu_el") where el is the name of the element.
            ax (plt.Axes): Axes object to plot on. If None, will create a new plot.
            annotate (bool): Whether to annotate each "phase" with the label of
                the entry. If no label, uses the reduced formula
            show_unphysical_only (bool): Whether to only show the shaded region where
                surface energy is negative. Useful for drawing other chempot range maps.
            fontsize (int): Font size of the annotation
        """
    delu_dict = delu_dict or {}
    ax = ax or pretty_plot(12, 8)
    el1, el2 = (str(elements[0]), str(elements[1]))
    delu1 = Symbol(f'delu_{elements[0]}')
    delu2 = Symbol(f'delu_{elements[1]}')
    range1 = ranges[0]
    range2 = ranges[1]
    vertices_dict: dict[SlabEntry, list] = {}
    for dmu1 in np.linspace(range1[0], range1[1], incr):
        new_delu_dict = delu_dict.copy()
        new_delu_dict[delu1] = dmu1
        range_dict, se_dict = self.stable_u_range_dict(range2, delu2, dmu_at_0=True, miller_index=miller_index, no_doped=no_doped, no_clean=no_clean, delu_dict=new_delu_dict, return_se_dict=True)
        for entry, vertex in range_dict.items():
            if not vertex:
                continue
            vertices_dict.setdefault(entry, [])
            selist = se_dict[entry]
            vertices_dict[entry].append({delu1: dmu1, delu2: [vertex, selist]})
    for entry, vertex in vertices_dict.items():
        xvals, yvals = ([], [])
        for ii, pt1 in enumerate(vertex):
            if len(pt1[delu2][1]) == 3:
                if pt1[delu2][1][0] < 0:
                    neg_dmu_range = [pt1[delu2][0][0], pt1[delu2][0][1]]
                else:
                    neg_dmu_range = [pt1[delu2][0][1], pt1[delu2][0][2]]
                ax.plot([pt1[delu1], pt1[delu1]], neg_dmu_range, 'k--')
            elif pt1[delu2][1][0] < 0 and pt1[delu2][1][1] < 0 and (not show_unphysical_only):
                ax.plot([pt1[delu1], pt1[delu1]], range2, 'k--')
            if ii == len(vertex) - 1:
                break
            pt2 = vertex[ii + 1]
            if not show_unphysical_only:
                ax.plot([pt1[delu1], pt2[delu1]], [pt1[delu2][0][0], pt2[delu2][0][0]], 'k')
            xvals.extend([pt1[delu1], pt2[delu1]])
            yvals.extend([pt1[delu2][0][0], pt2[delu2][0][0]])
        pt = vertex[-1]
        delu1, delu2 = pt
        xvals.extend([pt[delu1], pt[delu1]])
        yvals.extend(pt[delu2][0])
        if not show_unphysical_only:
            ax.plot([pt[delu1], pt[delu1]], [pt[delu2][0][0], pt[delu2][0][-1]], 'k')
        if annotate:
            x = np.mean([max(xvals), min(xvals)])
            y = np.mean([max(yvals), min(yvals)])
            label = entry.label or entry.reduced_formula
            ax.annotate(label, xy=[x, y], xytext=[x, y], fontsize=fontsize)
    ax.set(xlim=range1, ylim=range2)
    ax.set_xlabel(f'$\\Delta\\mu_{{{el1}}} (eV)$', fontsize=25)
    ax.set_ylabel(f'$\\Delta\\mu_{{{el2}}} (eV)$', fontsize=25)
    ax.set_xticks(rotation=60)
    return ax