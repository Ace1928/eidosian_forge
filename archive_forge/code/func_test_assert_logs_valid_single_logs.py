import logging
import warnings
import pytest
import cirq.testing
def test_assert_logs_valid_single_logs():
    with cirq.testing.assert_logs('apple'):
        logging.error('orange apple fruit')
    with cirq.testing.assert_logs('apple', 'orange'):
        logging.error('orange apple fruit')
    with cirq.testing.assert_logs():
        logging.error('orange apple fruit')
    with cirq.testing.assert_logs('apple', 'fruit'):
        logging.error('orange apple fruit')
    with cirq.testing.assert_logs('apple') as logs:
        logging.error('orange apple fruit')
    assert len(logs) == 1
    assert logs[0].getMessage() == 'orange apple fruit'
    assert logs[0].levelno == logging.ERROR
    with cirq.testing.assert_logs('apple'):
        warnings.warn('orange apple fruit')