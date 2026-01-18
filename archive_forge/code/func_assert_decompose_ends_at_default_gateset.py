from typing import Any
import numpy as np
from cirq import devices, protocols, ops, circuits
from cirq.testing import lin_alg_utils
def assert_decompose_ends_at_default_gateset(val: Any, ignore_known_gates: bool=True):
    """Asserts that cirq.decompose(val) ends at default cirq gateset or a known gate."""
    args = () if isinstance(val, ops.Operation) else (tuple(devices.LineQid.for_gate(val)),)
    dec_once = protocols.decompose_once(val, [val(*args[0]) if args else val], *args)
    for op in [*ops.flatten_to_ops((protocols.decompose(d) for d in dec_once))]:
        assert _known_gate_with_no_decomposition(op.gate) and ignore_known_gates or op in protocols.decompose_protocol.DECOMPOSE_TARGET_GATESET, f'{val} decomposed to {op}, which is not part of default cirq target gateset.'