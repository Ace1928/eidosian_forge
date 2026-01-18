from typing import Optional, Callable, List, Union
from functools import reduce
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library.standard_gates import HGate
from ..n_local.n_local import NLocal
def pauli_evolution(self, pauli_string, time):
    """Get the evolution block for the given pauli string."""
    pauli_string = pauli_string[::-1]
    trimmed = []
    indices = []
    for i, pauli in enumerate(pauli_string):
        if pauli != 'I':
            trimmed += [pauli]
            indices += [i]
    evo = QuantumCircuit(len(pauli_string))
    if len(trimmed) == 0:
        return evo

    def basis_change(circuit, inverse=False):
        for i, pauli in enumerate(pauli_string):
            if pauli == 'X':
                circuit.h(i)
            elif pauli == 'Y':
                circuit.rx(-np.pi / 2 if inverse else np.pi / 2, i)

    def cx_chain(circuit, inverse=False):
        num_cx = len(indices) - 1
        for i in reversed(range(num_cx)) if inverse else range(num_cx):
            circuit.cx(indices[i], indices[i + 1])
    basis_change(evo)
    cx_chain(evo)
    evo.p(self.alpha * time, indices[-1])
    cx_chain(evo, inverse=True)
    basis_change(evo, inverse=True)
    return evo