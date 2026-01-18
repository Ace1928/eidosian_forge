import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
def test_measure_at_end_invert_mask():
    simulator = cirq.Simulator()
    a = cirq.NamedQubit('a')
    circuit = cirq.Circuit(cirq.measure(a, key='a', invert_mask=(True,)))
    result = simulator.run(circuit, repetitions=4)
    np.testing.assert_equal(result.measurements['a'], np.array([[1]] * 4))