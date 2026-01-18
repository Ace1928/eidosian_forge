import cirq
def decompose_func(op: cirq.Operation, _):
    return cirq.decompose_once(op, context=cirq.DecompositionContext(qm)) if isinstance(op.gate, gate_type) else op