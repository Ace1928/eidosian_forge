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
def freq_units(units: Literal['thz', 'ev', 'mev', 'ha', 'cm-1', 'cm^-1']) -> FreqUnits:
    """
    Args:
        units: str, accepted values: thz, ev, mev, ha, cm-1, cm^-1.

    Returns:
        Conversion factor from THz to the required units and the label in the form of a namedtuple
    """
    dct = {'thz': FreqUnits(1, 'THz'), 'ev': FreqUnits(const.value('hertz-electron volt relationship') * const.tera, 'eV'), 'mev': FreqUnits(const.value('hertz-electron volt relationship') * const.tera / const.milli, 'meV'), 'ha': FreqUnits(const.value('hertz-hartree relationship') * const.tera, 'Ha'), 'cm-1': FreqUnits(const.value('hertz-inverse meter relationship') * const.tera * const.centi, 'cm^{-1}'), 'cm^-1': FreqUnits(const.value('hertz-inverse meter relationship') * const.tera * const.centi, 'cm^{-1}')}
    try:
        return dct[units.lower().strip()]
    except KeyError:
        raise KeyError(f'Value for units `{units}` unknown\nPossible values are:\n {list(dct)}')