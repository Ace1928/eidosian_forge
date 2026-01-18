import itertools
import numpy as np
import pytest
import cirq
import cirq.testing
def test_factor_validation():
    args = cirq.DensityMatrixSimulator()._create_simulation_state(0, qubits=cirq.LineQubit.range(2))
    args.apply_operation(cirq.H(cirq.LineQubit(0)))
    t = args.create_merged_state().target_tensor
    cirq.linalg.transformations.factor_density_matrix(t, [0])
    cirq.linalg.transformations.factor_density_matrix(t, [1])
    args.apply_operation(cirq.CNOT(cirq.LineQubit(0), cirq.LineQubit(1)))
    t = args.create_merged_state().target_tensor
    with pytest.raises(ValueError, match='factor'):
        cirq.linalg.transformations.factor_density_matrix(t, [0])
    with pytest.raises(ValueError, match='factor'):
        cirq.linalg.transformations.factor_density_matrix(t, [1])