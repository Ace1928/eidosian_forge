from __future__ import annotations
import contextlib
from typing import Union, Iterable, Any, Tuple, Optional, List, Literal, TYPE_CHECKING
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.circuit.classical import expr, types
from qiskit.circuit.exceptions import CircuitError
from .builder import InstructionPlaceholder, InstructionResources, ControlFlowBuilderBlock
from .control_flow import ControlFlowOp
from ._builder_utils import unify_circuit_resources, partition_registers, node_resources
def cases_specifier(self) -> Iterable[Tuple[Tuple, QuantumCircuit]]:
    """Return an iterable where each element is a 2-tuple whose first element is a tuple of
        jump values, and whose second is the single circuit block that is associated with those
        values.

        This is an abstract specification of the jump table suitable for creating new
        :class:`.SwitchCaseOp` instances.

        .. seealso::
            :meth:`.SwitchCaseOp.cases`
                Create a lookup table that you can use for your own purposes to jump from values to
                the circuit that would be executed."""
    return zip(self._label_spec, self._params)