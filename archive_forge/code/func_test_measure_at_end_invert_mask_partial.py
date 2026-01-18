import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
def test_measure_at_end_invert_mask_partial():
    simulator = cirq.Simulator()
    a, _, c = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.measure(a, c, key='ac', invert_mask=(True,)))
    result = simulator.run(circuit, repetitions=4)
    np.testing.assert_equal(result.measurements['ac'], np.array([[1, 0]] * 4))