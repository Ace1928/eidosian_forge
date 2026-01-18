import math
from unittest import mock
import pytest
from cirq_google.line.placement import optimization
def test_anneal_minimize_calls_trace_func():
    trace_func = mock.Mock()
    optimization.anneal_minimize('initial', lambda s: 1.0 if s == 'initial' else 0.0, lambda s: 'better', lambda: 1.0, 1.0, 0.5, 0.5, 1, trace_func=trace_func)
    trace_func.assert_has_calls([mock.call('initial', 1.0, 1.0, 1.0, True), mock.call('better', 1.0, 0.0, 1.0, True)])