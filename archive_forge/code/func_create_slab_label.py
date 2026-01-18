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
@property
def create_slab_label(self):
    """Returns a label (str) for this particular slab based on composition, coverage and Miller index."""
    if 'label' in self.data:
        return self.data['label']
    label = str(self.miller_index)
    ads_strs = list(self.ads_entries_dict)
    cleaned = self.cleaned_up_slab
    label += f' {cleaned.composition.reduced_composition}'
    if self.adsorbates:
        for ads in ads_strs:
            label += f'+{ads}'
        label += f', {self.get_monolayer:.3f} ML'
    return label