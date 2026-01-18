import abc
import copy
from typing import (
from typing_extensions import Self
import numpy as np
from cirq import ops, protocols, value
from cirq.sim.simulation_state_base import SimulationStateBase
@property
def prng(self) -> np.random.RandomState:
    return self._prng