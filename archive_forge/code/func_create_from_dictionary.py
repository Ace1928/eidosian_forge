from typing import (
import numpy as np
import cirq
from cirq import value
from cirq_google.calibration.phased_fsim import (
@classmethod
def create_from_dictionary(cls, parameters: Dict[Tuple[cirq.Qid, cirq.Qid], Dict[cirq.FSimGate, Union[PhasedFSimCharacterization, Dict]]], *, simulator: Optional[cirq.Simulator]=None) -> 'PhasedFSimEngineSimulator':
    """Creates PhasedFSimEngineSimulator with fixed drifts.

        Args:
            parameters: maps every pair of qubits and engine gate on that pair to a
                characterization for that gate.
            simulator: Simulator object to use. When None, a new instance of cirq.Simulator() will
                be created.

        Returns:
            New PhasedFSimEngineSimulator instance.

        Raises:
            ValueError: If missing parameters for the given pair of qubits.
        """
    for a, b in parameters.keys():
        if a > b:
            raise ValueError(f'All qubit pairs must be given in canonical order where the first qubit is less than the second, got {a} > {b}')

    def sample_gate(a: cirq.Qid, b: cirq.Qid, gate: cirq.FSimGate) -> PhasedFSimCharacterization:
        pair_parameters = None
        swapped = False
        if (a, b) in parameters:
            pair_parameters = parameters[a, b].get(gate)
        elif (b, a) in parameters:
            pair_parameters = parameters[b, a].get(gate)
            swapped = True
        if pair_parameters is None:
            raise ValueError(f'Missing parameters for value for pair {(a, b)} and gate {gate}.')
        if not isinstance(pair_parameters, PhasedFSimCharacterization):
            pair_parameters = PhasedFSimCharacterization(**pair_parameters)
        if swapped:
            pair_parameters = pair_parameters.parameters_for_qubits_swapped()
        return pair_parameters
    if simulator is None:
        simulator = cirq.Simulator()
    return cls(simulator, drift_generator=sample_gate, gates_translator=try_convert_gate_to_fsim)