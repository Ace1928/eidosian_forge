from __future__ import annotations
from typing import Optional, Union, Type
from math import ceil, pi
import numpy
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.singleton import SingletonGate, SingletonControlledGate, stdlib_singleton_key
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit._utils import _ctrl_state_to_int, with_gate_array, with_controlled_gate_array
@property
def num_ancilla_qubits(self):
    """The number of ancilla qubits."""
    return self.__class__.get_num_ancilla_qubits(self.num_ctrl_qubits)