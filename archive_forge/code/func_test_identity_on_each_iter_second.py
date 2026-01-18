import itertools
from typing import Any
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
def test_identity_on_each_iter_second():

    class Q(cirq.Qid):

        @property
        def dimension(self) -> int:
            return 2

        def _comparison_key(self) -> Any:
            return 1

        def __iter__(self):
            raise NotImplementedError()
    q = Q()
    assert cirq.I.on_each(q) == [cirq.I(q)]