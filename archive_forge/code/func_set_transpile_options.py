from __future__ import annotations
import math
from collections.abc import Sequence
from typing import Any
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.providers.backend import BackendV1, BackendV2
from qiskit.providers.options import Options
from qiskit.result import QuasiDistribution, Result
from qiskit.transpiler.passmanager import PassManager
from .backend_estimator import _prepare_counts, _run_circuits
from .base import BaseSampler, SamplerResult
from .primitive_job import PrimitiveJob
from .utils import _circuit_key
def set_transpile_options(self, **fields):
    """Set the transpiler options for transpiler.
        Args:
            **fields: The fields to update the options.
        Returns:
            self.
        Raises:
            QiskitError: if the instance has been closed.
        """
    self._transpile_options.update_options(**fields)