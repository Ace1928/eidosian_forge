from functools import partial
import numpy as np
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes.synthesis.plugin import UnitarySynthesisPlugin
@property
def supports_gate_lengths(self):
    """The plugin does not support gate lengths."""
    return False