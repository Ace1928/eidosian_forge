import dataclasses
import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union
import numpy as np
import quimb.tensor as qtn
from cirq import devices, protocols, qis, value
from cirq.sim import simulator_base
from cirq.sim.simulation_state import SimulationState
def i_str(self, i: int) -> str:
    return self._state.i_str(i)