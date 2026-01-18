from __future__ import annotations
import functools
import itertools
import re
from typing import Literal
import numpy as np
from qiskit.circuit import Instruction, QuantumCircuit
from qiskit.circuit.library.standard_gates import HGate, IGate, SGate, XGate, YGate, ZGate
from qiskit.circuit.operation import Operation
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.mixins import AdjointMixin, generate_apidocs
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.scalar_op import ScalarOp
from qiskit.quantum_info.operators.symplectic.base_pauli import _count_y
from .base_pauli import BasePauli
from .clifford_circuits import _append_circuit, _append_operation
@destab_z.setter
def destab_z(self, value):
    self.tableau[:self.num_qubits, self.num_qubits:2 * self.num_qubits] = value