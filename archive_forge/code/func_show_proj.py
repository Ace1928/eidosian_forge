from __future__ import annotations
import logging
from collections import namedtuple
from typing import TYPE_CHECKING, Callable
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from matplotlib.collections import LineCollection
from monty.json import jsanitize
from pymatgen.electronic_structure.plotter import BSDOSPlotter, plot_brillouin_zone
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.gruneisen import GruneisenPhononBandStructureSymmLine
from pymatgen.util.plotting import add_fig_kwargs, get_ax_fig, pretty_plot
def show_proj(self, site_comb: str | list[list[int]]='element', ylim: tuple[None | float, None | float] | None=None, units: Literal['thz', 'ev', 'mev', 'ha', 'cm-1', 'cm^-1']='thz', rgb_labels: tuple[str] | None=None) -> None:
    """Show the projected plot using matplotlib.

        Args:
            site_comb: A list of list of indices of sites to combine. For example,
                [[0, 1], [2, 3]] will combine the projections of sites 0 and 1,
                and sites 2 and 3. Defaults to "element", which will combine
                sites by element.
            ylim: Specify the y-axis (frequency) limits; by default None let
                the code choose.
            units: units for the frequencies. Accepted values thz, ev, mev, ha, cm-1, cm^-1.
                Defaults to "thz".
            rgb_labels: A list of labels for the rgb triangle. Defaults to None,
                which will use the element symbols.
        """
    self.get_proj_plot(site_comb=site_comb, ylim=ylim, units=units, rgb_labels=rgb_labels)
    plt.show()