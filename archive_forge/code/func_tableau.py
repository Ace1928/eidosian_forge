from typing import Optional, Sequence, TYPE_CHECKING
import numpy as np
from cirq.qis import clifford_tableau
from cirq.sim.clifford.stabilizer_simulation_state import StabilizerSimulationState
@property
def tableau(self) -> 'cirq.CliffordTableau':
    return self.state