from typing import Any, Dict, Iterable, List, Optional, Sequence, Union
from collections import defaultdict
import itertools
import random
import numpy as np
import sympy
from cirq import circuits, ops, linalg, protocols, qis
from cirq.testing import lin_alg_utils
def assert_has_consistent_apply_unitary(val: Any, *, atol: float=1e-08) -> None:
    """Tests whether a value's _apply_unitary_ is correct.

    Contrasts the effects of the value's `_apply_unitary_` with the
    matrix returned by the value's `_unitary_` method.

    Args:
        val: The value under test. Should have a `__pow__` method.
        atol: Absolute error tolerance.
    """
    __tracebackhide__ = True
    _assert_apply_unitary_works_when_axes_transposed(val, atol=atol)
    expected = protocols.unitary(val, default=None)
    qid_shape = protocols.qid_shape(val)
    eye = qis.eye_tensor((2,) + qid_shape, dtype=np.complex128)
    actual = protocols.apply_unitary(unitary_value=val, args=protocols.ApplyUnitaryArgs(target_tensor=eye, available_buffer=np.ones_like(eye) * float('nan'), axes=list(range(1, len(qid_shape) + 1))), default=None)
    if expected is None:
        assert actual is None
    else:
        expected = np.kron(np.eye(2), expected)
    if actual is not None:
        assert expected is not None
        n = np.prod([2, *qid_shape])
        np.testing.assert_allclose(actual.reshape(n, n), expected, atol=atol)