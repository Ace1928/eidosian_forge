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
def build_coupling_map(self, two_q_gate=None, filter_idle_qubits=False):
    """Get a :class:`~qiskit.transpiler.CouplingMap` from this target.

        If there is a mix of two qubit operations that have a connectivity
        constraint and those that are globally defined this will also return
        ``None`` because the globally connectivity means there is no constraint
        on the target. If you wish to see the constraints of the two qubit
        operations that have constraints you should use the ``two_q_gate``
        argument to limit the output to the gates which have a constraint.

        Args:
            two_q_gate (str): An optional gate name for a two qubit gate in
                the ``Target`` to generate the coupling map for. If specified the
                output coupling map will only have edges between qubits where
                this gate is present.
            filter_idle_qubits (bool): If set to ``True`` the output :class:`~.CouplingMap`
                will remove any qubits that don't have any operations defined in the
                target. Note that using this argument will result in an output
                :class:`~.CouplingMap` object which has holes in its indices
                which might differ from the assumptions of the class. The typical use
                case of this argument is to be paired with
                :meth:`.CouplingMap.connected_components` which will handle the holes
                as expected.
        Returns:
            CouplingMap: The :class:`~qiskit.transpiler.CouplingMap` object
                for this target. If there are no connectivity constraints in
                the target this will return ``None``.

        Raises:
            ValueError: If a non-two qubit gate is passed in for ``two_q_gate``.
            IndexError: If an Instruction not in the ``Target`` is passed in for
                ``two_q_gate``.
        """
    if self.qargs is None:
        return None
    if None not in self.qargs and any((len(x) > 2 for x in self.qargs)):
        logger.warning('This Target object contains multiqubit gates that operate on > 2 qubits. This will not be reflected in the output coupling map.')
    if two_q_gate is not None:
        coupling_graph = rx.PyDiGraph(multigraph=False)
        coupling_graph.add_nodes_from([None] * self.num_qubits)
        for qargs, properties in self._gate_map[two_q_gate].items():
            if len(qargs) != 2:
                raise ValueError('Specified two_q_gate: %s is not a 2 qubit instruction' % two_q_gate)
            coupling_graph.add_edge(*qargs, {two_q_gate: properties})
        cmap = CouplingMap()
        cmap.graph = coupling_graph
        return cmap
    if self._coupling_graph is None:
        self._build_coupling_graph()
    if self._coupling_graph is not None:
        cmap = CouplingMap()
        if filter_idle_qubits:
            cmap.graph = self._filter_coupling_graph()
        else:
            cmap.graph = self._coupling_graph.copy()
        return cmap
    else:
        return None