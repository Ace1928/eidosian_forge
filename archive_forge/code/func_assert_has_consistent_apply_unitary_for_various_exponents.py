from typing import Any, Dict, Iterable, List, Optional, Sequence, Union
from collections import defaultdict
import itertools
import random
import numpy as np
import sympy
from cirq import circuits, ops, linalg, protocols, qis
from cirq.testing import lin_alg_utils
def assert_has_consistent_apply_unitary_for_various_exponents(val: Any, *, exponents=(0, 1, -1, 0.5, 0.25, -0.5, 0.1, sympy.Symbol('s'))) -> None:
    """Tests whether a value's _apply_unitary_ is correct.

    Contrasts the effects of the value's `_apply_unitary_` with the
    matrix returned by the value's `_unitary_` method. Attempts this after
    attempting to raise the value to several exponents.

    Args:
        val: The value under test. Should have a `__pow__` method.
        exponents: The exponents to try. Defaults to a variety of special and
            arbitrary angles, as well as a parameterized angle (a symbol). If
            the value's `__pow__` returns `NotImplemented` for any of these,
            they are skipped.
    """
    __tracebackhide__ = True
    for exponent in exponents:
        gate = protocols.pow(val, exponent, default=None)
        if gate is not None:
            assert_has_consistent_apply_unitary(gate)