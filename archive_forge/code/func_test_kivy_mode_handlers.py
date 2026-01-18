import logging
import os
import pathlib
import sys
import time
import pytest
@pytest.mark.skipif(LOG_MODE != 'KIVY', reason='Requires KIVY_LOG_MODE==KIVY to run.')
def test_kivy_mode_handlers():
    assert are_regular_logs_handled()
    assert are_kivy_logger_logs_handled()