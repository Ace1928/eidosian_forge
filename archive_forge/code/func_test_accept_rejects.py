import math
from unittest import mock
import pytest
from cirq_google.line.placement import optimization
def test_accept_rejects():
    assert not optimization._accept(1.0 - 1e-09, 1.0, 1.0)[0]
    assert not optimization._accept(1.0 / math.e + 1e-09, 1.0, 1.0)[0]