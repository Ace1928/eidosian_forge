import dataclasses
import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union
import numpy as np
import quimb.tensor as qtn
from cirq import devices, protocols, qis, value
from cirq.sim import simulator_base
from cirq.sim.simulation_state import SimulationState
def estimation_stats(self):
    """Returns some statistics about the memory usage and quality of the approximation."""
    return self._state.estimation_stats()