import numpy as np
import ase.units as u
from ase.vibrations.raman import Raman, RamanPhonons
from ase.vibrations.resonant_raman import ResonantRaman
from ase.calculators.excitation_list import polarizability
def set_approximation(self, value):
    approx = value.lower()
    if approx in ['profeta', 'placzek', 'p-p']:
        self._approx = value
    else:
        raise ValueError('Please use "Profeta", "Placzek" or "P-P".')