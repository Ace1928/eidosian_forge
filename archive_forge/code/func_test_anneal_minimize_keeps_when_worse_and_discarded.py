import math
from unittest import mock
import pytest
from cirq_google.line.placement import optimization
def test_anneal_minimize_keeps_when_worse_and_discarded():
    assert optimization.anneal_minimize('initial', lambda s: 0.0 if s == 'initial' else 1.0, lambda s: 'better', lambda: 0.9, 1.0, 0.5, 0.5, 1) == 'initial'