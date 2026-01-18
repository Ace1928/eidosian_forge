import os
import sys
import pytest
from nipype import config
from unittest.mock import MagicMock
def test_display_empty_patched(monkeypatch):
    """
    Check that when $DISPLAY is empty string and no option is specified,
    a virtual Xvfb is used
    """
    config._display = None
    if config.has_option('execution', 'display_variable'):
        config._config.remove_option('execution', 'display_variable')
    monkeypatch.setenv('DISPLAY', '')
    monkeypatch.setitem(sys.modules, 'xvfbwrapper', xvfbpatch)
    monkeypatch.setattr(sys, 'platform', value='linux')
    assert config.get_display() == ':2010'
    assert config.get_display() == ':2010'
    config._display = None
    monkeypatch.setattr(sys, 'platform', value='darwin')
    with pytest.raises(RuntimeError):
        config.get_display()