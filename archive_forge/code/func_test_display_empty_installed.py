import os
import sys
import pytest
from nipype import config
from unittest.mock import MagicMock
@pytest.mark.skipif(not has_Xvfb, reason='xvfbwrapper not installed')
@pytest.mark.skipif('darwin' in sys.platform, reason='macosx requires root for Xvfb')
def test_display_empty_installed(monkeypatch):
    """
    Check that actually uses xvfbwrapper when installed (not mocked)
    and necessary (no config and $DISPLAY empty)
    """
    config._display = None
    if config.has_option('execution', 'display_variable'):
        config._config.remove_option('execution', 'display_variable')
    monkeypatch.setenv('DISPLAY', '')
    newdisp = config.get_display()
    assert int(newdisp.split(':')[-1]) > 1000
    assert config.get_display() == newdisp