from typing import Tuple, Any, List, Union, Dict
import pytest
import cirq
from pyquil import Program
import numpy as np
import sympy
from cirq_rigetti import circuit_sweep_executors as executors, circuit_transformers
def test_invalid_pyquil_region_measurement(mock_qpu_implementer: Any, parametric_circuit_with_params: Tuple[cirq.Circuit, cirq.Sweepable]) -> None:
    """test that executors raise `ValueError` if the measurement_id_map
    does not exist.
    """
    parametric_circuit, sweepable = parametric_circuit_with_params
    repetitions = 2
    param_resolvers = [r for r in cirq.to_resolvers(sweepable)]
    expected_results = [np.ones((repetitions,)) * (params['t'] if 't' in params else i) for i, params in enumerate(param_resolvers)]
    quantum_computer = mock_qpu_implementer.implement_passive_quantum_computer_with_results(expected_results)

    def broken_hook(program: Program, measurement_id_map: Dict[str, str]) -> Tuple[Program, Dict[str, str]]:
        return (program, {cirq_key: f'{cirq_key}-doesnt-exist' for cirq_key in measurement_id_map})
    transformer = circuit_transformers.build(post_transformation_hooks=[broken_hook])
    with pytest.raises(ValueError):
        _ = executors.with_quilc_compilation_and_cirq_parameter_resolution(transformer=transformer, quantum_computer=quantum_computer, circuit=parametric_circuit, resolvers=param_resolvers, repetitions=repetitions)