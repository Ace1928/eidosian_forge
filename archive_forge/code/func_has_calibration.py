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
def has_calibration(self, operation_name: str, qargs: tuple[int, ...]) -> bool:
    """Return whether the instruction (operation + qubits) defines a calibration.

        Args:
            operation_name: The name of the operation for the instruction.
            qargs: The tuple of qubit indices for the instruction.

        Returns:
            Returns ``True`` if the calibration is supported and ``False`` if it isn't.
        """
    qargs = tuple(qargs)
    if operation_name not in self._gate_map:
        return False
    if qargs not in self._gate_map[operation_name]:
        return False
    return getattr(self._gate_map[operation_name][qargs], '_calibration') is not None