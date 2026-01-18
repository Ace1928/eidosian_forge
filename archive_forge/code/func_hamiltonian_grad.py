import pennylane as qml
def hamiltonian_grad(tape, idx):
    """Computes the tapes necessary to get the gradient of a tape with respect to
    a Hamiltonian observable's coefficients.

    Args:
        tape (qml.tape.QuantumTape): tape with a single Hamiltonian expectation as measurement
        idx (int): index of parameter that we differentiate with respect to
    """
    op, m_pos, p_idx = tape.get_operation(idx)
    queue_position = m_pos - len(tape.operations)
    new_measurements = list(tape.measurements)
    new_measurements[queue_position] = qml.expval(op.ops[p_idx])
    new_tape = qml.tape.QuantumScript(tape.operations, new_measurements, shots=tape.shots)
    if len(tape.measurements) > 1:

        def processing_fn(results):
            res = results[0][queue_position]
            zeros = qml.math.zeros_like(res)
            final = []
            for i, _ in enumerate(tape.measurements):
                final.append(res if i == queue_position else zeros)
            return qml.math.expand_dims(qml.math.stack(final), 0)
        return ([new_tape], processing_fn)
    return ([new_tape], lambda x: x)