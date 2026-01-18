import logging
import os
import pathlib
import sys
import time
import pytest
@pytest.mark.skipif(LOG_MODE != 'KIVY', reason='Requires KIVY_LOG_MODE==KIVY to run.')
def test_kivy_log_mode_marker_off():
    """
    This is a test of the pytest markers.
    This should only be invoked if the environment variable is properly set
    (before pytest is run).

    Also, tests that kivy.logger paid attention to the environment variable
    """
    assert logging.root.level == 0, 'Root logger was not modified'