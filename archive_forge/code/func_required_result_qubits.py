from typing import Union, Optional, List
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister, ParameterExpression
from ..basis_change import QFT
@staticmethod
def required_result_qubits(quadratic: Union[np.ndarray, List[List[float]]], linear: Union[np.ndarray, List[float]], offset: float) -> int:
    """Get the number of required result qubits.

        Args:
            quadratic: A matrix containing the quadratic coefficients.
            linear: An array containing the linear coefficients.
            offset: A constant offset.

        Returns:
            The number of qubits needed to represent the value of the quadratic form
            in twos complement.
        """
    bounds = []
    for condition in [lambda x: x < 0, lambda x: x > 0]:
        bound = 0.0
        bound += sum((sum((q_ij for q_ij in q_i if condition(q_ij))) for q_i in quadratic))
        bound += sum((l_i for l_i in linear if condition(l_i)))
        bound += offset if condition(offset) else 0
        bounds.append(bound)
    num_qubits_for_min = int(np.ceil(np.log2(max(-bounds[0], 1))))
    num_qubits_for_max = int(np.ceil(np.log2(bounds[1] + 1)))
    num_result_qubits = 1 + max(num_qubits_for_min, num_qubits_for_max)
    return num_result_qubits