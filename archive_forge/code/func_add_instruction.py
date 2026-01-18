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
def add_instruction(self, instruction, properties=None, name=None):
    """Add a new instruction to the :class:`~qiskit.transpiler.Target`

        As ``Target`` objects are strictly additive this is the primary method
        for modifying a ``Target``. Typically, you will use this to fully populate
        a ``Target`` before using it in :class:`~qiskit.providers.BackendV2`. For
        example::

            from qiskit.circuit.library import CXGate
            from qiskit.transpiler import Target, InstructionProperties

            target = Target()
            cx_properties = {
                (0, 1): None,
                (1, 0): None,
                (0, 2): None,
                (2, 0): None,
                (0, 3): None,
                (2, 3): None,
                (3, 0): None,
                (3, 2): None
            }
            target.add_instruction(CXGate(), cx_properties)

        Will add a :class:`~qiskit.circuit.library.CXGate` to the target with no
        properties (duration, error, etc) with the coupling edge list:
        ``(0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (2, 3), (3, 0), (3, 2)``. If
        there are properties available for the instruction you can replace the
        ``None`` value in the properties dictionary with an
        :class:`~qiskit.transpiler.InstructionProperties` object. This pattern
        is repeated for each :class:`~qiskit.circuit.Instruction` the target
        supports.

        Args:
            instruction (Union[qiskit.circuit.Instruction, Type[qiskit.circuit.Instruction]]):
                The operation object to add to the map. If it's parameterized any value
                of the parameter can be set. Optionally for variable width
                instructions (such as control flow operations such as :class:`~.ForLoop` or
                :class:`~MCXGate`) you can specify the class. If the class is specified than the
                ``name`` argument must be specified. When a class is used the gate is treated as global
                and not having any properties set.
            properties (dict): A dictionary of qarg entries to an
                :class:`~qiskit.transpiler.InstructionProperties` object for that
                instruction implementation on the backend. Properties are optional
                for any instruction implementation, if there are no
                :class:`~qiskit.transpiler.InstructionProperties` available for the
                backend the value can be None. If there are no constraints on the
                instruction (as in a noiseless/ideal simulation) this can be set to
                ``{None, None}`` which will indicate it runs on all qubits (or all
                available permutations of qubits for multi-qubit gates). The first
                ``None`` indicates it applies to all qubits and the second ``None``
                indicates there are no
                :class:`~qiskit.transpiler.InstructionProperties` for the
                instruction. By default, if properties is not set it is equivalent to
                passing ``{None: None}``.
            name (str): An optional name to use for identifying the instruction. If not
                specified the :attr:`~qiskit.circuit.Instruction.name` attribute
                of ``gate`` will be used. All gates in the ``Target`` need unique
                names. Backends can differentiate between different
                parameterization of a single gate by providing a unique name for
                each (e.g. `"rx30"`, `"rx60", ``"rx90"`` similar to the example in the
                documentation for the :class:`~qiskit.transpiler.Target` class).
        Raises:
            AttributeError: If gate is already in map
            TranspilerError: If an operation class is passed in for ``instruction`` and no name
                is specified or ``properties`` is set.
        """
    is_class = inspect.isclass(instruction)
    if not is_class:
        instruction_name = name or instruction.name
    else:
        if not name:
            raise TranspilerError('A name must be specified when defining a supported global operation by class')
        if properties is not None:
            raise TranspilerError("An instruction added globally by class can't have properties set.")
        instruction_name = name
    if properties is None:
        properties = {None: None}
    if instruction_name in self._gate_map:
        raise AttributeError('Instruction %s is already in the target' % instruction_name)
    self._gate_name_map[instruction_name] = instruction
    if is_class:
        qargs_val = {None: None}
    else:
        if None in properties:
            self._global_operations[instruction.num_qubits].add(instruction_name)
        qargs_val = {}
        for qarg in properties:
            if qarg is not None and len(qarg) != instruction.num_qubits:
                raise TranspilerError(f'The number of qubits for {instruction} does not match the number of qubits in the properties dictionary: {qarg}')
            if qarg is not None:
                self.num_qubits = max(self.num_qubits, max(qarg) + 1)
            qargs_val[qarg] = properties[qarg]
            self._qarg_gate_map[qarg].add(instruction_name)
    self._gate_map[instruction_name] = qargs_val
    self._coupling_graph = None
    self._instruction_durations = None
    self._instruction_schedule_map = None
    self._non_global_basis = None
    self._non_global_strict_basis = None