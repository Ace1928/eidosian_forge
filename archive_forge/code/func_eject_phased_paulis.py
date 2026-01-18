from typing import Optional, cast, TYPE_CHECKING, Iterable, Tuple, Dict
import sympy
import numpy as np
from cirq import circuits, ops, value, protocols
from cirq.transformers import transformer_api, transformer_primitives
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
@transformer_api.transformer(add_deep_support=True)
def eject_phased_paulis(circuit: 'cirq.AbstractCircuit', *, context: Optional['cirq.TransformerContext']=None, atol: float=1e-08, eject_parameterized: bool=False) -> 'cirq.Circuit':
    """Transformer pass to push X, Y, PhasedX & (certain) PhasedXZ gates to the end of the circuit.

    As the gates get pushed, they may absorb Z gates, cancel against other
    X, Y, or PhasedX gates with exponent=1, get merged into measurements (as
    output bit flips), and cause phase kickback operations across CZs (which can
    then be removed by the `cirq.eject_z` transformation).

    `cirq.PhasedXZGate` with `z_exponent=0` (i.e. equivalent to PhasedXPow) or with `x_exponent=0`
    and `axis_phase_exponent=0` (i.e. equivalent to ZPowGate) are also supported.
    To eject `PhasedXZGates` with arbitrary x/z/axis exponents, run
    `cirq.eject_z(cirq.eject_phased_paulis(cirq.eject_z(circuit)))`.

    Args:
        circuit: Input circuit to transform.
        context: `cirq.TransformerContext` storing common configurable options for transformers.
        atol: Maximum absolute error tolerance. The optimization is permitted to simply drop
            negligible combinations gates with a threshold determined by this tolerance.
        eject_parameterized: If True, the optimization will attempt to eject parameterized gates
            as well.  This may result in other gates parameterized by symbolic expressions.
    Returns:
          Copy of the transformed input circuit.
    """
    held_w_phases: Dict[ops.Qid, value.TParamVal] = {}
    tags_to_ignore = set(context.tags_to_ignore) if context else set()

    def map_func(op: 'cirq.Operation', _: int) -> 'cirq.OP_TREE':
        if set(op.tags) & tags_to_ignore:
            return [_dump_held(op.qubits, held_w_phases), op]
        w = _try_get_known_phased_pauli(op, no_symbolic=not eject_parameterized)
        if w is not None:
            return _potential_cross_whole_w(op, atol, held_w_phases) if single_qubit_decompositions.is_negligible_turn((w[0] - 1) / 2, atol) else _potential_cross_partial_w(op, held_w_phases)
        affected = [q for q in op.qubits if q in held_w_phases]
        if not affected:
            return op
        t = _try_get_known_z_half_turns(op, no_symbolic=not eject_parameterized)
        if t is not None:
            return _absorb_z_into_w(op, held_w_phases)
        if isinstance(op.gate, ops.MeasurementGate):
            return _dump_into_measurement(op, held_w_phases)
        if _try_get_known_cz_half_turns(op, no_symbolic=not eject_parameterized) is not None:
            return _single_cross_over_cz(op, affected[0]) if len(affected) == 1 else _double_cross_over_cz(op, held_w_phases)
        return [_dump_held(op.qubits, held_w_phases), op]
    return circuits.Circuit(transformer_primitives.map_operations_and_unroll(circuit, map_func), _dump_held(held_w_phases.keys(), held_w_phases))