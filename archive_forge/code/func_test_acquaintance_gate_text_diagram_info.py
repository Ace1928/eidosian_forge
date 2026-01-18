from itertools import combinations, product
from random import randint
from string import ascii_lowercase as alphabet
from typing import Optional, Sequence, Tuple
import numpy
import pytest
import cirq
import cirq.testing as ct
import cirq.contrib.acquaintance as cca
def test_acquaintance_gate_text_diagram_info():
    qubits = [cirq.NamedQubit(s) for s in 'xyz']
    circuit = cirq.Circuit([cirq.Moment([cca.acquaint(*qubits)])])
    actual_text_diagram = circuit.to_text_diagram().strip()
    expected_text_diagram = '\nx: ───█───\n      │\ny: ───█───\n      │\nz: ───█───\n    '.strip()
    assert actual_text_diagram == expected_text_diagram