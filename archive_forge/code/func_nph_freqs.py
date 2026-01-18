from __future__ import annotations
import json
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.core.spectrum import Spectrum
from pymatgen.core.structure import Structure
from pymatgen.util.plotting import add_fig_kwargs
from pymatgen.vis.plotters import SpectrumPlotter
@property
def nph_freqs(self) -> int:
    """Number of phonon frequencies."""
    return len(self.ph_freqs_gamma)