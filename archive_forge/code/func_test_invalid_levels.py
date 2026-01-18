import logging
import warnings
import pytest
import cirq.testing
def test_invalid_levels():
    with pytest.raises(ValueError, match='min_level.*max_level'):
        with cirq.testing.assert_logs('test', min_level=logging.CRITICAL, max_level=logging.WARNING):
            pass