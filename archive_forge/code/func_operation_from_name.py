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
def operation_from_name(self, instruction):
    """Get the operation class object for a given name

        Args:
            instruction (str): The instruction name to get the
                :class:`~qiskit.circuit.Instruction` instance for
        Returns:
            qiskit.circuit.Instruction: The Instruction instance corresponding to the
            name. This also can also be the class for globally defined variable with
            operations.
        """
    return self._gate_name_map[instruction]