import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
def test_entangled_reset_does_not_break_randomness():
    """Test for bad assumptions on caching the wave function on general channels.

    A previous version of cirq made the mistake of assuming that it was okay to
    cache the wavefunction produced by general channels on unrelated qubits
    before repeatedly sampling measurements. This test checks for that mistake.
    """
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(a), cirq.CNOT(a, b), cirq.ResetChannel().on(a), cirq.measure(b, key='out'))
    samples = cirq.Simulator().sample(circuit, repetitions=100)['out']
    counts = samples.value_counts()
    assert len(counts) == 2
    assert 10 <= counts[0] <= 90
    assert 10 <= counts[1] <= 90