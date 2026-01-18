import numpy as np
from ase.calculators.calculator import Calculator
from ase.data import atomic_numbers
from ase.utils import IOContext
from ase.geometry import get_distances
from ase.cell import Cell
def combine_lj(self):
    self.sigma, self.epsilon = combine_lj_lorenz_berthelot(self.sigmaqm, self.sigmamm, self.epsilonqm, self.epsilonmm)