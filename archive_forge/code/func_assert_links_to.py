import pytest
import sympy
import cirq
from cirq.contrib.quirk.export_to_quirk import circuit_to_quirk_url
def assert_links_to(circuit: cirq.Circuit, expected: str, **kwargs):
    actual = circuit_to_quirk_url(circuit, **kwargs)
    actual = actual.replace('\n', '').replace(' ', '').strip()
    expected = expected.replace('],[', '],\n[').strip()
    expected = expected.replace('\n', '').replace(' ', '')
    assert actual == expected