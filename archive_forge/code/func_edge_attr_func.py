from rustworkx.visualization import graphviz_draw
from qiskit.dagcircuit.dagnode import DAGOpNode, DAGInNode, DAGOutNode
from qiskit.circuit import Qubit, Clbit, ClassicalRegister
from qiskit.circuit.classical import expr
from qiskit.converters import dagdependency_to_circuit
from qiskit.utils import optionals as _optionals
from qiskit.exceptions import InvalidFileError
from .exceptions import VisualizationError
def edge_attr_func(edge):
    e = {}
    if isinstance(edge, Qubit):
        label = register_bit_labels.get(edge, f'q_{dag.find_bit(edge).index}')
    else:
        label = register_bit_labels.get(edge, f'c_{dag.find_bit(edge).index}')
    e['label'] = label
    return e