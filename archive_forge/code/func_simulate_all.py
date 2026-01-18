import numbers
import warnings
import numpy as np
from .kalman_smoother import KalmanSmoother
from .cfa_simulation_smoother import CFASimulationSmoother
from . import tools
@simulate_all.setter
def simulate_all(self, value):
    if bool(value):
        self.simulation_output = self.simulation_output | SIMULATION_ALL
    else:
        self.simulation_output = self.simulation_output & ~SIMULATION_ALL