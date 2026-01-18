from __future__ import annotations
import json
import os
import warnings
from typing import Literal
import matplotlib.pyplot as plt
import numpy as np
from monty.json import MSONable
from pandas import DataFrame
from plotly.graph_objects import Figure, Scatter
from pymatgen.analysis.phase_diagram import GrandPotentialPhaseDiagram, PhaseDiagram
from pymatgen.analysis.reaction_calculator import Reaction
from pymatgen.core.composition import Composition
from pymatgen.util.due import Doi, due
from pymatgen.util.plotting import pretty_plot
from pymatgen.util.string import htmlify, latexify
@classmethod
def get_chempot_correction(cls, element: str, temp: float, pres: float):
    """
        Get the normalized correction term Δμ for chemical potential of a gas
        phase consisting of element at given temperature and pressure,
        referenced to that in the standard state (T_std = 298.15 K,
        T_std = 1 bar). The gas phase is limited to be one of O2, N2, Cl2,
        F2, H2. Calculation formula can be found in the documentation of
        Materials Project website.

        Args:
            element: The string representing the element.
            temp: The temperature of the gas phase in Kelvin.
            pres: The pressure of the gas phase in Pa.

        Returns:
            The correction of chemical potential in eV/atom of the gas
            phase at given temperature and pressure.
        """
    if element not in ['O', 'N', 'Cl', 'F', 'H']:
        warnings.warn(f"element={element!r} not one of valid options: ['O', 'N', 'Cl', 'F', 'H']")
        return 0
    std_temp = 298.15
    std_pres = 100000.0
    ideal_gas_const = 8.3144598
    cp_dict = {'O': 29.376, 'N': 29.124, 'Cl': 33.949, 'F': 31.302, 'H': 28.836}
    s_dict = {'O': 205.147, 'N': 191.609, 'Cl': 223.079, 'F': 202.789, 'H': 130.68}
    cp_std = cp_dict[element]
    s_std = s_dict[element]
    pv_correction = ideal_gas_const * temp * np.log(pres / std_pres)
    ts_correction = -cp_std * (temp * np.log(temp) - std_temp * np.log(std_temp)) + cp_std * (temp - std_temp) * (1 + np.log(std_temp)) - s_std * (temp - std_temp)
    dg = pv_correction + ts_correction
    dg /= 1000 * cls.EV_TO_KJ_PER_MOL
    dg /= 2
    return dg