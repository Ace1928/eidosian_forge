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
def target_to_backend_properties(target: Target):
    """Convert a :class:`~.Target` object into a legacy :class:`~.BackendProperties`"""
    properties_dict: dict[str, Any] = {'backend_name': '', 'backend_version': '', 'last_update_date': None, 'general': []}
    gates = []
    qubits = []
    for gate, qargs_list in target.items():
        if gate != 'measure':
            for qargs, props in qargs_list.items():
                property_list = []
                if getattr(props, 'duration', None) is not None:
                    property_list.append({'date': datetime.datetime.now(datetime.timezone.utc), 'name': 'gate_length', 'unit': 's', 'value': props.duration})
                if getattr(props, 'error', None) is not None:
                    property_list.append({'date': datetime.datetime.now(datetime.timezone.utc), 'name': 'gate_error', 'unit': '', 'value': props.error})
                if property_list:
                    gates.append({'gate': gate, 'qubits': list(qargs), 'parameters': property_list, 'name': gate + '_'.join([str(x) for x in qargs])})
        else:
            qubit_props: dict[int, Any] = {}
            if target.num_qubits is not None:
                qubit_props = {x: None for x in range(target.num_qubits)}
            for qargs, props in qargs_list.items():
                if qargs is None:
                    continue
                qubit = qargs[0]
                props_list = []
                if getattr(props, 'error', None) is not None:
                    props_list.append({'date': datetime.datetime.now(datetime.timezone.utc), 'name': 'readout_error', 'unit': '', 'value': props.error})
                if getattr(props, 'duration', None) is not None:
                    props_list.append({'date': datetime.datetime.now(datetime.timezone.utc), 'name': 'readout_length', 'unit': 's', 'value': props.duration})
                if not props_list:
                    qubit_props = {}
                    break
                qubit_props[qubit] = props_list
            if qubit_props and all((x is not None for x in qubit_props.values())):
                qubits = [qubit_props[i] for i in range(target.num_qubits)]
    if gates or qubits:
        properties_dict['gates'] = gates
        properties_dict['qubits'] = qubits
        return BackendProperties.from_dict(properties_dict)
    else:
        return None