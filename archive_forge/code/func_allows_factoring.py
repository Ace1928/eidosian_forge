import abc
import copy
from typing import (
from typing_extensions import Self
import numpy as np
from cirq import ops, protocols, value
from cirq.sim.simulation_state_base import SimulationStateBase
@property
def allows_factoring(self):
    """Subclasses that allow factorization should override this."""
    return self._state.supports_factor if self._state is not None else False