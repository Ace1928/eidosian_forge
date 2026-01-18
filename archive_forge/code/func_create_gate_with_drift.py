from typing import (
import numpy as np
import cirq
from cirq import value
from cirq_google.calibration.phased_fsim import (
def create_gate_with_drift(self, a: cirq.Qid, b: cirq.Qid, gate_calibration: PhaseCalibratedFSimGate) -> cirq.PhasedFSimGate:
    """Generates a gate with drift for a given gate.

        Args:
            a: The first qubit.
            b: The second qubit.
            gate_calibration: Reference gate together with a phase information.

        Returns:
            A modified gate that includes the drifts induced by internal state of the simulator.
        """
    gate = gate_calibration.engine_gate
    if (a, b, gate) in self._drifted_parameters:
        parameters = self._drifted_parameters[a, b, gate]
    elif (b, a, gate) in self._drifted_parameters:
        parameters = self._drifted_parameters[b, a, gate].parameters_for_qubits_swapped()
    else:
        parameters = self._drift_generator(a, b, gate)
        self._drifted_parameters[a, b, gate] = parameters
    return gate_calibration.as_characterized_phased_fsim_gate(parameters)