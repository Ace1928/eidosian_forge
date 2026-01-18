from typing import Tuple, List, cast
import re
import pytest
import sympy
import cirq
from cirq._compat import proper_repr
import numpy as np
@pytest.mark.parametrize('gate, tags_to_accept, tags_to_ignore', [(CustomX, [], []), (CustomX, ['tag1'], []), (CustomX, [], ['tag2']), (CustomX, ['tag3'], ['tag4']), (CustomXPowGate, [], [])])
def test_gate_family_default_name_and_description(gate, tags_to_accept, tags_to_ignore):
    g = cirq.GateFamily(gate, tags_to_accept=tags_to_accept, tags_to_ignore=tags_to_ignore)
    assert re.match('.*GateFamily.*CustomX.*', g.name)
    assert re.match('Accepts.*instances.*CustomX.*', g.description)
    accepted_match = re.compile('.*Accepted tags.*', re.DOTALL).match(g.description)
    assert (accepted_match is None) == (tags_to_accept == [])
    ignored_match = re.compile('.*Ignored tags.*', re.DOTALL).match(g.description)
    assert (ignored_match is None) == (tags_to_ignore == [])