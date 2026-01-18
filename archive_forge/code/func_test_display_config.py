import os
import sys
import pytest
from nipype import config
from unittest.mock import MagicMock
@pytest.mark.parametrize('dispnum', range(5))
def test_display_config(monkeypatch, dispnum):
    """Check that the display_variable option is used ($DISPLAY not set)"""
    config._display = None
    dispstr = ':%d' % dispnum
    config.set('execution', 'display_variable', dispstr)
    monkeypatch.delitem(os.environ, 'DISPLAY', raising=False)
    assert config.get_display() == config.get('execution', 'display_variable')
    assert config.get_display() == config.get('execution', 'display_variable')