from __future__ import annotations
import itertools
import logging
import re
import warnings
from copy import deepcopy
from functools import cmp_to_key, partial
from multiprocessing import Pool
from typing import TYPE_CHECKING, Any, no_type_check
import numpy as np
from monty.json import MontyDecoder, MSONable
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.special import comb
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.analysis.reaction_calculator import Reaction, ReactionError
from pymatgen.core import Composition, Element
from pymatgen.core.ion import Ion
from pymatgen.entries.compatibility import MU_H2O
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.util.coord import Simplex
from pymatgen.util.due import Doi, due
from pymatgen.util.plotting import pretty_plot
from pymatgen.util.string import Stringify
@no_type_check
def plot_entry_stability(self, entry: Any, pH_range: tuple[float, float]=(-2, 16), pH_resolution: int=100, V_range: tuple[float, float]=(-3, 3), V_resolution: int=100, e_hull_max: float=1, cmap: str='RdYlBu_r', ax: plt.Axes | None=None, **kwargs: Any) -> plt.Axes:
    """
        Plots the stability of an entry in the Pourbaix diagram.

        Args:
            entry (Any): The entry to plot stability for.
            pH_range (tuple[float, float], optional): pH range for the plot. Defaults to (-2, 16).
            pH_resolution (int, optional): pH resolution. Defaults to 100.
            V_range (tuple[float, float], optional): Voltage range for the plot. Defaults to (-3, 3).
            V_resolution (int, optional): Voltage resolution. Defaults to 100.
            e_hull_max (float, optional): Maximum energy above the hull. Defaults to 1.
            cmap (str, optional): Colormap for the plot. Defaults to "RdYlBu_r".
            ax (Axes, optional): Existing matplotlib Axes object for plotting. Defaults to None.
            **kwargs (Any): Additional keyword arguments passed to `get_pourbaix_plot`.

        Returns:
            plt.Axes: Matplotlib Axes object with the plotted stability.
        """
    ax = self.get_pourbaix_plot(ax=ax, **kwargs)
    pH, V = np.mgrid[pH_range[0]:pH_range[1]:pH_resolution * 1j, V_range[0]:V_range[1]:V_resolution * 1j]
    stability = self._pbx.get_decomposition_energy(entry, pH, V)
    cax = ax.pcolor(pH, V, stability, cmap=cmap, vmin=0, vmax=e_hull_max)
    cbar = ax.figure.colorbar(cax)
    cbar.set_label(f'Stability of {generate_entry_label(entry)} (eV/atom)')
    return ax