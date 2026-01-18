import dataclasses
import cirq
import cirq_google
import pytest
from cirq_google import (
def test_equality_from_dictionaries():
    d1 = {'a': 1, 'b': 2}
    d2 = {'b': 2, 'a': 1}
    assert d1 == d2
    k1 = KeyValueExecutableSpec.from_dict(d1, executable_family='test')
    k2 = KeyValueExecutableSpec.from_dict(d2, executable_family='test')
    assert k1 == k2