from math import pi
from typing import Optional, Union, Tuple, List
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister, Qubit
from qiskit.circuit.library.standard_gates.x import MCXGate
from qiskit.circuit.library.standard_gates.u3 import _generate_gray_code
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.exceptions import QiskitError
def mcrz(self, lam: ParameterValueType, q_controls: Union[QuantumRegister, List[Qubit]], q_target: Qubit, use_basis_gates: bool=False):
    """
    Apply Multiple-Controlled Z rotation gate

    Args:
        self (QuantumCircuit): The QuantumCircuit object to apply the mcrz gate on.
        lam (float): angle lambda
        q_controls (list(Qubit)): The list of control qubits
        q_target (Qubit): The target qubit
        use_basis_gates (bool): use p, u, cx

    Raises:
        QiskitError: parameter errors
    """
    from .rz import CRZGate, RZGate
    control_qubits = self.qbit_argument_conversion(q_controls)
    target_qubit = self.qbit_argument_conversion(q_target)
    if len(target_qubit) != 1:
        raise QiskitError('The mcrz gate needs a single qubit as target.')
    all_qubits = control_qubits + target_qubit
    target_qubit = target_qubit[0]
    self._check_dups(all_qubits)
    n_c = len(control_qubits)
    if n_c == 1:
        if use_basis_gates:
            self.u(0, 0, lam / 2, target_qubit)
            self.cx(control_qubits[0], target_qubit)
            self.u(0, 0, -lam / 2, target_qubit)
            self.cx(control_qubits[0], target_qubit)
        else:
            self.append(CRZGate(lam), control_qubits + [target_qubit])
    else:
        cgate = _mcsu2_real_diagonal(RZGate(lam).to_matrix(), num_controls=len(control_qubits), use_basis_gates=use_basis_gates)
        self.compose(cgate, control_qubits + [target_qubit], inplace=True)