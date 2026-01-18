import numbers
import warnings
import numpy as np
from .kalman_smoother import KalmanSmoother
from .cfa_simulation_smoother import CFASimulationSmoother
from . import tools
def simulator(self, nsimulations, random_state=None):
    return self.simulation_smoother(simulation_output=0, method='kfs', nobs=nsimulations, random_state=random_state)