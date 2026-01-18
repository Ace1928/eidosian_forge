from qiskit.circuit import QuantumCircuit, CircuitInstruction, ClassicalRegister
from qiskit.circuit.controlflow import condition_resources
from . import DAGOpNode, DAGCircuit, DAGDependency
from .exceptions import DAGCircuitError
def not_filter_fn(node):
    """Returns the opposite of filter_fn."""
    return not filter_fn(node)