import logging
import warnings
import pytest
import cirq.testing
def test_assert_logs_invalid_single_logs():
    match = "^dog expected to appear in log messages but it was not found. Log messages: \\['orange apple fruit'\\].$"
    with pytest.raises(AssertionError, match=match):
        with cirq.testing.assert_logs('dog'):
            logging.error('orange apple fruit')
    with pytest.raises(AssertionError, match='dog'):
        with cirq.testing.assert_logs('dog', 'cat'):
            logging.error('orange apple fruit')