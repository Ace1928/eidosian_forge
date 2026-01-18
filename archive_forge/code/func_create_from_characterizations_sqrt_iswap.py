from typing import (
import numpy as np
import cirq
from cirq import value
from cirq_google.calibration.phased_fsim import (
@classmethod
def create_from_characterizations_sqrt_iswap(cls, characterizations: Iterable[PhasedFSimCalibrationResult], *, simulator: Optional[cirq.Simulator]=None, ideal_when_missing_gate: bool=False, ideal_when_missing_parameter: bool=False) -> 'PhasedFSimEngineSimulator':
    """Creates PhasedFSimEngineSimulator with fixed drifts from the characterizations results.

        Args:
            characterizations: Characterization results which are source of the parameters for
                each gate.
            simulator: Simulator object to use. When None, a new instance of cirq.Simulator() will
                be created.
            ideal_when_missing_gate: When set and parameters for some gate for a given pair of
                qubits are not specified in the parameters dictionary then the
                FSimGate(theta=π/4, phi=0) gate parameters will be used. When not set and this
                situation occurs, ValueError is thrown during simulation.
            ideal_when_missing_parameter: When set and some parameter for some gate for a given pair
                of qubits is specified then the matching parameter of FSimGate(theta=π/4, phi=0)
                gate will be used. When not set and this situation occurs, ValueError is thrown
                during simulation.

        Returns:
            New PhasedFSimEngineSimulator instance.

        Raises:
            ValueError: If the gate was not a gate like `ISWAP ** -0.5` or the pair of qubits it
                acts on appears in multiple different moments.
        """
    parameters: PhasedFsimDictParameters = {}
    for characterization in characterizations:
        gate = characterization.gate
        _assert_inv_sqrt_iswap_like(gate)
        for (a, b), pair_parameters in characterization.parameters.items():
            if a > b:
                a, b = (b, a)
                pair_parameters = pair_parameters.parameters_for_qubits_swapped()
            if (a, b) in parameters:
                raise ValueError(f'Pair ({(a, b)}) appears in multiple moments, multi-moment simulation is not supported.')
            parameters[a, b] = pair_parameters
    if simulator is None:
        simulator = cirq.Simulator()
    return cls.create_from_dictionary_sqrt_iswap(parameters, simulator=simulator, ideal_when_missing_gate=ideal_when_missing_gate, ideal_when_missing_parameter=ideal_when_missing_parameter)