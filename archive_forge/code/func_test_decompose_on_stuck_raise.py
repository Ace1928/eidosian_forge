import itertools
from typing import Optional
from unittest import mock
import pytest
import cirq
def test_decompose_on_stuck_raise():
    a, b = cirq.LineQubit.range(2)
    no_method = NoMethod()
    with pytest.raises(ValueError, match="but can't be decomposed"):
        _ = cirq.decompose(NoMethod(), keep=lambda _: False)
    assert cirq.decompose([], keep=lambda _: False) == []
    assert cirq.decompose([], on_stuck_raise=None) == []
    assert cirq.decompose(no_method, keep=lambda _: False, on_stuck_raise=None) == [no_method]
    assert cirq.decompose(no_method, keep=lambda _: False, on_stuck_raise=lambda _: None) == [no_method]
    with pytest.raises(TypeError, match='test'):
        _ = cirq.decompose(no_method, keep=lambda _: False, on_stuck_raise=TypeError('test'))
    with pytest.raises(NotImplementedError, match='op cirq.CZ'):
        _ = cirq.decompose(cirq.CZ(a, b), keep=lambda _: False, on_stuck_raise=lambda op: NotImplementedError(f'op {op!r}'))
    with pytest.raises(ValueError, match='on_stuck_raise'):
        assert cirq.decompose([], on_stuck_raise=TypeError('x'))