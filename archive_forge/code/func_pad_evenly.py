import abc
import collections
from typing import (
import numpy as np
from cirq import devices, ops, protocols, study, value
from cirq.sim.simulation_product_state import SimulationProductState
from cirq.sim.simulation_state import TSimulationState
from cirq.sim.simulation_state_base import SimulationStateBase
from cirq.sim.simulator import (
def pad_evenly(results: Sequence[Sequence[Sequence[int]]]):
    largest = max((len(result) for result in results))
    xs = np.zeros((len(results), largest, len(results[0][0])), dtype=np.uint8)
    for i, result in enumerate(results):
        xs[i, 0:len(result), :] = result
    return xs