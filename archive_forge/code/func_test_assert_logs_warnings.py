import logging
import warnings
import pytest
import cirq.testing
def test_assert_logs_warnings():
    with warnings.catch_warnings(record=True):
        with cirq.testing.assert_logs('apple'):
            warnings.warn('orange apple fruit')
        with cirq.testing.assert_logs('apple', count=2):
            warnings.warn('orange apple fruit')
            logging.error('other')
        with cirq.testing.assert_logs('apple', capture_warnings=False):
            logging.error('orange apple fruit')
            warnings.warn('warn')
        with pytest.raises(AssertionError, match='^Expected 1 log message but got 0. Log messages.*$'):
            with cirq.testing.assert_logs('apple', capture_warnings=False):
                warnings.warn('orange apple fruit')