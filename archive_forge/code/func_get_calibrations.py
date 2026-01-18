from typing import (
import numpy as np
import cirq
from cirq import value
from cirq_google.calibration.phased_fsim import (
def get_calibrations(self, requests: Sequence[PhasedFSimCalibrationRequest]) -> List[PhasedFSimCalibrationResult]:
    """Retrieves the calibration that matches the requests

        Args:
            requests: Calibration requests to obtain.

        Returns:
            Calibration results that reflect the internal state of simulator.

        Raises:
            ValueError: If supplied type of request is not supported or if the request contains
                and unsupported gate.
        """
    results = []
    for request in requests:
        if isinstance(request, FloquetPhasedFSimCalibrationRequest):
            options = request.options
            characterize_theta = options.characterize_theta
            characterize_zeta = options.characterize_zeta
            characterize_chi = options.characterize_chi
            characterize_gamma = options.characterize_gamma
            characterize_phi = options.characterize_phi
        else:
            raise ValueError(f'Unsupported calibration request {request}')
        translated = self.gates_translator(request.gate)
        if translated is None:
            raise ValueError(f'Calibration request contains unsupported gate {request.gate}')
        parameters = {}
        for a, b in request.pairs:
            drifted = self.create_gate_with_drift(a, b, translated)
            parameters[a, b] = PhasedFSimCharacterization(theta=cast(float, drifted.theta) if characterize_theta else None, zeta=cast(float, drifted.zeta) if characterize_zeta else None, chi=cast(float, drifted.chi) if characterize_chi else None, gamma=cast(float, drifted.gamma) if characterize_gamma else None, phi=cast(float, drifted.phi) if characterize_phi else None)
        results.append(PhasedFSimCalibrationResult(parameters=parameters, gate=request.gate, options=options))
    return results