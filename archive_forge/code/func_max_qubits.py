from functools import partial
import numpy as np
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes.synthesis.plugin import UnitarySynthesisPlugin
@property
def max_qubits(self):
    """Maximum number of supported qubits is ``14``."""
    return 14