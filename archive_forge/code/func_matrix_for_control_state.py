import numpy
from qiskit.exceptions import QiskitError
from qiskit.circuit.exceptions import CircuitError
from .parametervector import ParameterVectorElement
def matrix_for_control_state(state):
    out = numpy.asarray(_compute_control_matrix(base, num_ctrl_qubits, state), dtype=numpy.complex128)
    out.setflags(write=False)
    return out