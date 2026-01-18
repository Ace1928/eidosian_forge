from typing import List, Optional, Set, Tuple, TYPE_CHECKING
from cirq import protocols, circuits
from cirq.transformers import transformer_api
def find_terminal_measurements(circuit: 'cirq.AbstractCircuit') -> List[Tuple[int, 'cirq.Operation']]:
    """Finds all terminal measurements in the given circuit.

    A measurement is terminal if there are no other operations acting on the measured qubits
    after the measurement operation occurs in the circuit.

    Args:
        circuit: The circuit to find terminal measurements in.

    Returns:
        List of terminal measurements (unordered), each specified as
        (moment_index, measurement_operation).
    """
    open_qubits: Set['cirq.Qid'] = set(circuit.all_qubits())
    seen_control_keys: Set['cirq.MeasurementKey'] = set()
    terminal_measurements: Set[Tuple[int, 'cirq.Operation']] = set()
    for i in range(len(circuit) - 1, -1, -1):
        moment = circuit[i]
        for q in open_qubits:
            op = moment.operation_at(q)
            if op is not None and open_qubits.issuperset(op.qubits) and protocols.is_measurement(op) and (not seen_control_keys & protocols.measurement_key_objs(op)):
                terminal_measurements.add((i, op))
        open_qubits -= moment.qubits
        seen_control_keys |= protocols.control_keys(moment)
        if not open_qubits:
            break
    return list(terminal_measurements)