from copy import copy
import logging
from collections import deque
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit.library.standard_gates import CXGate, RZXGate
from qiskit.dagcircuit import DAGOpNode
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.optimization.optimize_1q_decomposition import (
def mov_list(destination, source):
    """
    Replace `destination` in-place with `source`.
    """
    while destination:
        del destination[0]
    destination += source