from functools import partial
import numpy as np
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes.synthesis.plugin import UnitarySynthesisPlugin
@property
def supports_natural_direction(self):
    """The plugin does not support natural direction,
        it assumes bidirectional two qubit gates."""
    return False