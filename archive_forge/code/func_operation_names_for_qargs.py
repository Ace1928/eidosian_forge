from a backend
from __future__ import annotations
import itertools
from typing import Optional, List, Any
from collections.abc import Mapping
from collections import defaultdict
import datetime
import io
import logging
import inspect
import rustworkx as rx
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.circuit.gate import Gate
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.pulse.calibration_entries import CalibrationEntry, ScheduleDef
from qiskit.pulse.schedule import Schedule, ScheduleBlock
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.instruction_durations import InstructionDurations
from qiskit.transpiler.timing_constraints import TimingConstraints
from qiskit.providers.exceptions import BackendPropertyError
from qiskit.pulse.exceptions import PulseError, UnassignedDurationError
from qiskit.exceptions import QiskitError
from qiskit.providers.backend import QubitProperties  # pylint: disable=unused-import
from qiskit.providers.models.backendproperties import BackendProperties
def operation_names_for_qargs(self, qargs):
    """Get the operation names for a specified qargs tuple

        Args:
            qargs (tuple): A ``qargs`` tuple of the qubits to get the gates that apply
                to it. For example, ``(0,)`` will return the set of all
                instructions that apply to qubit 0. If set to ``None`` this will
                return the names for any globally defined operations in the target.
        Returns:
            set: The set of operation names that apply to the specified ``qargs``.

        Raises:
            KeyError: If ``qargs`` is not in target
        """
    if self.num_qubits == 0 or self.num_qubits is None:
        qargs = None
    if qargs is not None and any((x not in range(0, self.num_qubits) for x in qargs)):
        raise KeyError(f'{qargs} not in target.')
    res = self._qarg_gate_map.get(qargs, set())
    if qargs is not None:
        res.update(self._global_operations.get(len(qargs), set()))
    for name, op in self._gate_name_map.items():
        if inspect.isclass(op):
            res.add(name)
    if not res:
        raise KeyError(f'{qargs} not in target.')
    return res