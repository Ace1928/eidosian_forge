from math import pi
from typing import Optional, Union, Tuple, List
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister, Qubit
from qiskit.circuit.library.standard_gates.x import MCXGate
from qiskit.circuit.library.standard_gates.u3 import _generate_gray_code
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.exceptions import QiskitError
def mcrx(self, theta: ParameterValueType, q_controls: Union[QuantumRegister, List[Qubit]], q_target: Qubit, use_basis_gates: bool=False):
    """
    Apply Multiple-Controlled X rotation gate

    Args:
        self (QuantumCircuit): The QuantumCircuit object to apply the mcrx gate on.
        theta (float): angle theta
        q_controls (QuantumRegister or list(Qubit)): The list of control qubits
        q_target (Qubit): The target qubit
        use_basis_gates (bool): use p, u, cx

    Raises:
        QiskitError: parameter errors
    """
    from .rx import RXGate
    control_qubits = self.qbit_argument_conversion(q_controls)
    target_qubit = self.qbit_argument_conversion(q_target)
    if len(target_qubit) != 1:
        raise QiskitError('The mcrz gate needs a single qubit as target.')
    all_qubits = control_qubits + target_qubit
    target_qubit = target_qubit[0]
    self._check_dups(all_qubits)
    n_c = len(control_qubits)
    if n_c == 1:
        _apply_cu(self, theta, -pi / 2, pi / 2, control_qubits[0], target_qubit, use_basis_gates=use_basis_gates)
    elif n_c < 4:
        theta_step = theta * (1 / 2 ** (n_c - 1))
        _apply_mcu_graycode(self, theta_step, -pi / 2, pi / 2, control_qubits, target_qubit, use_basis_gates=use_basis_gates)
    else:
        cgate = _mcsu2_real_diagonal(RXGate(theta).to_matrix(), num_controls=len(control_qubits), use_basis_gates=use_basis_gates)
        self.compose(cgate, control_qubits + [target_qubit], inplace=True)