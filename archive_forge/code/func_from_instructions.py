from __future__ import annotations
import copy
import multiprocessing as mp
import typing
from collections import OrderedDict, defaultdict, namedtuple
from typing import (
import numpy as np
from qiskit._accelerate.quantum_circuit import CircuitData
from qiskit.exceptions import QiskitError
from qiskit.utils.multiprocessing import is_main_process
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.exceptions import CircuitError
from . import _classical_resource_map
from ._utils import sort_parameters
from .controlflow.builder import CircuitScopeInterface, ControlFlowBuilderBlock
from .controlflow.break_loop import BreakLoopOp, BreakLoopPlaceholder
from .controlflow.continue_loop import ContinueLoopOp, ContinueLoopPlaceholder
from .controlflow.for_loop import ForLoopOp, ForLoopContext
from .controlflow.if_else import IfElseOp, IfContext
from .controlflow.switch_case import SwitchCaseOp, SwitchContext
from .controlflow.while_loop import WhileLoopOp, WhileLoopContext
from .classical import expr
from .parameterexpression import ParameterExpression, ParameterValueType
from .quantumregister import QuantumRegister, Qubit, AncillaRegister, AncillaQubit
from .classicalregister import ClassicalRegister, Clbit
from .parametertable import ParameterReferences, ParameterTable, ParameterView
from .parametervector import ParameterVector
from .instructionset import InstructionSet
from .operation import Operation
from .register import Register
from .bit import Bit
from .quantumcircuitdata import QuantumCircuitData, CircuitInstruction
from .delay import Delay
@staticmethod
def from_instructions(instructions: Iterable[CircuitInstruction | tuple[qiskit.circuit.Instruction] | tuple[qiskit.circuit.Instruction, Iterable[Qubit]] | tuple[qiskit.circuit.Instruction, Iterable[Qubit], Iterable[Clbit]]], *, qubits: Iterable[Qubit]=(), clbits: Iterable[Clbit]=(), name: str | None=None, global_phase: ParameterValueType=0, metadata: dict | None=None) -> 'QuantumCircuit':
    """Construct a circuit from an iterable of CircuitInstructions.

        Args:
            instructions: The instructions to add to the circuit.
            qubits: Any qubits to add to the circuit. This argument can be used,
                for example, to enforce a particular ordering of qubits.
            clbits: Any classical bits to add to the circuit. This argument can be used,
                for example, to enforce a particular ordering of classical bits.
            name: The name of the circuit.
            global_phase: The global phase of the circuit in radians.
            metadata: Arbitrary key value metadata to associate with the circuit.

        Returns:
            The quantum circuit.
        """
    circuit = QuantumCircuit(name=name, global_phase=global_phase, metadata=metadata)
    added_qubits = set()
    added_clbits = set()
    if qubits:
        qubits = list(qubits)
        circuit.add_bits(qubits)
        added_qubits.update(qubits)
    if clbits:
        clbits = list(clbits)
        circuit.add_bits(clbits)
        added_clbits.update(clbits)
    for instruction in instructions:
        if not isinstance(instruction, CircuitInstruction):
            instruction = CircuitInstruction(*instruction)
        qubits = [qubit for qubit in instruction.qubits if qubit not in added_qubits]
        clbits = [clbit for clbit in instruction.clbits if clbit not in added_clbits]
        circuit.add_bits(qubits)
        circuit.add_bits(clbits)
        added_qubits.update(qubits)
        added_clbits.update(clbits)
        circuit._append(instruction)
    return circuit