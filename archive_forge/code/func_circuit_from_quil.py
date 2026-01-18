from typing import Callable, cast, Dict, Union
import numpy as np
from pyquil.parser import parse
from pyquil.quilbase import (
from cirq import Circuit, LineQubit
from cirq.ops import (
def circuit_from_quil(quil: str) -> Circuit:
    """Convert a Quil program to a Cirq Circuit.

    Args:
        quil: The Quil program to convert.

    Returns:
        A Cirq Circuit generated from the Quil program.

    Raises:
        UnsupportedQuilInstruction: Cirq does not support the specified Quil instruction.
        UndefinedQuilGate: Cirq does not support the specified Quil gate.

    References:
        https://github.com/rigetti/pyquil
    """
    circuit = Circuit()
    defined_gates = SUPPORTED_GATES.copy()
    instructions = parse(quil)
    for inst in instructions:
        if isinstance(inst, DefGate):
            if inst.parameters:
                raise UnsupportedQuilInstruction('Parameterized DEFGATEs are currently unsupported.')
            defined_gates[inst.name] = MatrixGate(inst.matrix)
        elif isinstance(inst, Declare):
            pass
        elif isinstance(inst, PyQuilGate):
            quil_gate_name = inst.name
            quil_gate_params = inst.params
            line_qubits = list((LineQubit(q.index) for q in inst.qubits))
            if quil_gate_name not in defined_gates:
                raise UndefinedQuilGate(f'Quil gate {quil_gate_name} not supported in Cirq.')
            cirq_gate_fn = defined_gates[quil_gate_name]
            if quil_gate_params:
                circuit += cast(Callable[..., Gate], cirq_gate_fn)(*quil_gate_params)(*line_qubits)
            else:
                circuit += cirq_gate_fn(*line_qubits)
        elif isinstance(inst, PyQuilMeasurement):
            line_qubit = LineQubit(inst.qubit.index)
            if inst.classical_reg is None:
                raise UnsupportedQuilInstruction(f'Quil measurement {inst} without classical register not currently supported in Cirq.')
            quil_memory_reference = inst.classical_reg.out()
            circuit += MeasurementGate(1, key=quil_memory_reference)(line_qubit)
        elif isinstance(inst, Pragma):
            raise UnsupportedQuilInstruction(PRAGMA_ERROR)
        elif isinstance(inst, (Reset, ResetQubit)):
            raise UnsupportedQuilInstruction(RESET_ERROR)
        else:
            raise UnsupportedQuilInstruction(f'Quil instruction {inst} of type {type(inst)} not currently supported in Cirq.')
    return circuit