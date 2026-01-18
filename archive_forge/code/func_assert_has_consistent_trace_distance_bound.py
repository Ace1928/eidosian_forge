import itertools
from typing import Any, Dict, Optional, Sequence, Type, Union
import numpy as np
import sympy
from cirq import ops, protocols, value
from cirq.testing.consistent_act_on import assert_all_implemented_act_on_effects_match_unitary
from cirq.testing.circuit_compare import (
from cirq.testing.consistent_decomposition import (
from cirq.testing.consistent_phase_by import assert_phase_by_is_consistent_with_unitary
from cirq.testing.consistent_qasm import assert_qasm_is_consistent_with_unitary
from cirq.testing.consistent_pauli_expansion import (
from cirq.testing.consistent_resolve_parameters import assert_consistent_resolve_parameters
from cirq.testing.consistent_specified_has_unitary import assert_specifies_has_unitary_if_unitary
from cirq.testing.equivalent_repr_eval import assert_equivalent_repr
from cirq.testing.consistent_controlled_gate_op import (
from cirq.testing.consistent_unitary import assert_unitary_is_consistent
def assert_has_consistent_trace_distance_bound(val: Any) -> None:
    __tracebackhide__ = True
    u = protocols.unitary(val, default=None)
    val_from_trace = protocols.trace_distance_bound(val)
    assert 0.0 <= val_from_trace <= 1.0
    if u is not None:

        class Unitary:

            def _unitary_(self):
                return u
        val_from_unitary = protocols.trace_distance_bound(Unitary())
        assert val_from_trace >= val_from_unitary or np.isclose(val_from_trace, val_from_unitary)