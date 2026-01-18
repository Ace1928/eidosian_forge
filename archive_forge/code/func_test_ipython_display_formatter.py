from math import pi
import pytest
from IPython import get_ipython
from traitlets.config import Config
from IPython.core.formatters import (
from IPython.utils.io import capture_output
def test_ipython_display_formatter():
    """Objects with _ipython_display_ defined bypass other formatters"""
    f = get_ipython().display_formatter
    catcher = []

    class SelfDisplaying(object):

        def _ipython_display_(self):
            catcher.append(self)

    class NotSelfDisplaying(object):

        def __repr__(self):
            return 'NotSelfDisplaying'

        def _ipython_display_(self):
            raise NotImplementedError
    save_enabled = f.ipython_display_formatter.enabled
    f.ipython_display_formatter.enabled = True
    yes = SelfDisplaying()
    no = NotSelfDisplaying()
    d, md = f.format(no)
    assert d == {'text/plain': repr(no)}
    assert md == {}
    assert catcher == []
    d, md = f.format(yes)
    assert d == {}
    assert md == {}
    assert catcher == [yes]
    f.ipython_display_formatter.enabled = save_enabled