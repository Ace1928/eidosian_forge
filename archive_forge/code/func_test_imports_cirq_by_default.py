import numpy as np
import pytest
import cirq
def test_imports_cirq_by_default():
    cirq.testing.assert_equivalent_repr(cirq.NamedQubit('a'))