import logging
import warnings
import pytest
import cirq.testing
def test_assert_logs_valid_multiple_logs():
    with cirq.testing.assert_logs('apple', count=2):
        logging.error('orange apple fruit')
        logging.error('other')
    with cirq.testing.assert_logs('apple', count=2):
        logging.error('other')
        logging.error('orange apple fruit')
    with cirq.testing.assert_logs('apple', count=2):
        logging.error('other')
        logging.error('orange apple fruit')
    with cirq.testing.assert_logs('apple', count=2):
        logging.error('other')
        logging.error('orange apple fruit')
    with cirq.testing.assert_logs('apple', 'other', count=2):
        logging.error('other')
        logging.error('orange apple fruit')
    with cirq.testing.assert_logs('apple', count=3):
        logging.error('orange apple fruit')
        logging.error('other')
        logging.warning('other two')