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
def pd_coords(self, comp: Composition) -> np.ndarray:
    """
        The phase diagram is generated in a reduced dimensional space
        (n_elements - 1). This function returns the coordinates in that space.
        These coordinates are compatible with the stored simplex objects.

        Args:
            comp (Composition): A composition

        Returns:
            The coordinates for a given composition in the PhaseDiagram's basis
        """
    if set(comp.elements) - set(self.elements):
        raise ValueError(f'{comp} has elements not in the phase diagram {', '.join(map(str, self.elements))}')
    return np.array([comp.get_atomic_fraction(el) for el in self.elements[1:]])