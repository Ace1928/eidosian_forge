from functools import partial
import numpy as np
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes.synthesis.plugin import UnitarySynthesisPlugin
@property
def supports_gate_errors(self):
    """The plugin does not support gate errors."""
    return False