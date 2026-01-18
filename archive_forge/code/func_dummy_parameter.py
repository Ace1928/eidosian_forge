import collections
import copy
import itertools
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.dagcircuit.dagdependency import DAGDependency
from qiskit.converters.dagdependency_to_dag import dagdependency_to_dag
from qiskit.utils import optionals as _optionals
def dummy_parameter():
    return Parameter(f'_qiskit_template_dummy_{next(_dummy_counter)}')