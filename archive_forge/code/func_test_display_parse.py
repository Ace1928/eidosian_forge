import os
import sys
import pytest
from nipype import config
from unittest.mock import MagicMock
@pytest.mark.parametrize('dispvar', [':12', 'localhost:12', 'localhost:12.1'])
def test_display_parse(monkeypatch, dispvar):
    """Check that when $DISPLAY is defined, the display is correctly parsed"""
    config._display = None
    config._config.remove_option('execution', 'display_variable')
    monkeypatch.setenv('DISPLAY', dispvar)
    assert config.get_display() == ':12'
    assert config.get_display() == ':12'