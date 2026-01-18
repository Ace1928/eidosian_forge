from typing import Any, Dict, Iterable, List, Optional, Sequence, Union
from collections import defaultdict
import itertools
import random
import numpy as np
import sympy
from cirq import circuits, ops, linalg, protocols, qis
from cirq.testing import lin_alg_utils
def assert_has_consistent_apply_channel(val: Any, *, atol: float=1e-08) -> None:
    """Tests whether a value's _apply_channel_ is correct.

    Contrasts the effects of the value's `_apply_channel_` with the superoperator calculated from
    the Kraus components returned by the value's `_kraus_` method.

    Args:
        val: The value under test. Should have a `__pow__` method.
        atol: Absolute error tolerance.
    """
    __tracebackhide__ = True
    kraus = protocols.kraus(val, default=None)
    expected = qis.kraus_to_superoperator(kraus) if kraus is not None else None
    qid_shape = protocols.qid_shape(val)
    eye = qis.eye_tensor(qid_shape * 2, dtype=np.complex128)
    actual = protocols.apply_channel(val=val, args=protocols.ApplyChannelArgs(target_tensor=eye, out_buffer=np.ones_like(eye) * float('nan'), auxiliary_buffer0=np.ones_like(eye) * float('nan'), auxiliary_buffer1=np.ones_like(eye) * float('nan'), left_axes=list(range(len(qid_shape))), right_axes=list(range(len(qid_shape), len(qid_shape) * 2))), default=None)
    if expected is None:
        assert actual is None
    if actual is not None:
        assert expected is not None
        n = np.prod(qid_shape) ** 2
        np.testing.assert_allclose(actual.reshape((n, n)), expected, atol=atol)