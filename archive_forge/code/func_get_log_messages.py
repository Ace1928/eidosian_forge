import collections
import logging
import pytest
import modin.logging
from modin.config import LogMode
@pytest.fixture
def get_log_messages():
    old = LogMode.get()
    LogMode.enable()
    modin.logging.get_logger()
    yield _FakeLogger.get
    _FakeLogger.clear()
    LogMode.put(old)