import numpy as np
import pytest
import cirq
def test_inconclusive():

    class No:
        pass
    assert not cirq.has_unitary(object())
    assert not cirq.has_unitary('boo')
    assert not cirq.has_unitary(No())