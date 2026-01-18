import math
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import pytest
import sympy
import cirq
def test_measurements_retained_in_step_results():
    sim = SplittableCountingSimulator()
    circuit = cirq.Circuit(cirq.measure(q0, key='a'), cirq.measure(q0, key='b'), cirq.measure(q0, key='c'))
    iterator = sim.simulate_moment_steps(circuit)
    assert next(iterator).measurements.keys() == {'a'}
    assert next(iterator).measurements.keys() == {'a', 'b'}
    assert next(iterator).measurements.keys() == {'a', 'b', 'c'}
    assert not any(iterator)