import pytest
import logging
from charset_normalizer.utils import set_logging_handler
from charset_normalizer.api import from_bytes, explain_handler
from charset_normalizer.constant import TRACE
def test_explain_true_behavior(self, caplog):
    test_sequence = b'This is a test sequence of bytes that should be sufficient'
    from_bytes(test_sequence, steps=1, chunk_size=50, explain=True)
    assert explain_handler not in self.logger.handlers
    for record in caplog.records:
        assert record.levelname in ['Level 5', 'DEBUG']