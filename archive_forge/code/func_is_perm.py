from __future__ import annotations
import abc
import collections
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from pymatgen.core.spectrum import Spectrum
from pymatgen.util.plotting import add_fig_kwargs, pretty_plot
def is_perm(hkl1, hkl2) -> bool:
    h1 = np.abs(hkl1)
    h2 = np.abs(hkl2)
    return all((i == j for i, j in zip(sorted(h1), sorted(h2))))