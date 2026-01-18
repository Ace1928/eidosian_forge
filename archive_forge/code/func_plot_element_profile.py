from __future__ import annotations
import collections
import itertools
import json
import logging
import math
import os
import re
import warnings
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal, no_type_check
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib import cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.font_manager import FontProperties
from monty.json import MontyDecoder, MSONable
from scipy import interpolate
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from tqdm import tqdm
from pymatgen.analysis.reaction_calculator import Reaction, ReactionError
from pymatgen.core import DummySpecies, Element, get_el_sp
from pymatgen.core.composition import Composition
from pymatgen.entries import Entry
from pymatgen.util.coord import Simplex, in_coord_list
from pymatgen.util.due import Doi, due
from pymatgen.util.plotting import pretty_plot
from pymatgen.util.string import htmlify, latexify
def plot_element_profile(self, element, comp, show_label_index=None, xlim=5):
    """
        Draw the element profile plot for a composition varying different
        chemical potential of an element.

        X value is the negative value of the chemical potential reference to
        elemental chemical potential. For example, if choose Element("Li"),
        X= -(µLi-µLi0), which corresponds to the voltage versus metal anode.
        Y values represent for the number of element uptake in this composition
        (unit: per atom). All reactions are printed to help choosing the
        profile steps you want to show label in the plot.

        Args:
            element (Element): An element of which the chemical potential is
                considered. It also must be in the phase diagram.
            comp (Composition): A composition.
            show_label_index (list of integers): The labels for reaction products
                you want to show in the plot. Default to None (not showing any
                annotation for reaction products). For the profile steps you want
                to show the labels, just add it to the show_label_index. The
                profile step counts from zero. For example, you can set
                show_label_index=[0, 2, 5] to label profile step 0,2,5.
            xlim (float): The max x value. x value is from 0 to xlim. Default to
                5 eV.

        Returns:
            Plot of element profile evolution by varying the chemical potential
            of an element.
        """
    ax = pretty_plot(12, 8)
    pd = self._pd
    evolution = pd.get_element_profile(element, comp)
    num_atoms = evolution[0]['reaction'].reactants[0].num_atoms
    element_energy = evolution[0]['chempot']
    x1, x2, y1 = (None, None, None)
    for idx, dct in enumerate(evolution):
        v = -(dct['chempot'] - element_energy)
        if idx != 0:
            ax.plot([x2, x2], [y1, dct['evolution'] / num_atoms], 'k', linewidth=2.5)
        x1 = v
        y1 = dct['evolution'] / num_atoms
        x2 = -(evolution[idx + 1]['chempot'] - element_energy) if idx != len(evolution) - 1 else 5.0
        if show_label_index is not None and idx in show_label_index:
            products = [re.sub('(\\d+)', '$_{\\1}$', p.reduced_formula) for p in dct['reaction'].products if p.reduced_formula != element.symbol]
            ax.annotate(', '.join(products), xy=(v + 0.05, y1 + 0.05), fontsize=24, color='r')
            ax.plot([x1, x2], [y1, y1], 'r', linewidth=3)
        else:
            ax.plot([x1, x2], [y1, y1], 'k', linewidth=2.5)
    ax.set_xlim((0, xlim))
    ax.set_xlabel('-$\\Delta{\\mu}$ (eV)')
    ax.set_ylabel('Uptake per atom')
    return ax