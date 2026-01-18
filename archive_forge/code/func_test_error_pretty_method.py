from math import pi
import pytest
from IPython import get_ipython
from traitlets.config import Config
from IPython.core.formatters import (
from IPython.utils.io import capture_output
def test_error_pretty_method():
    f = PlainTextFormatter()

    class BadPretty(object):

        def _repr_pretty_(self):
            return 'hello'
    bad = BadPretty()
    with capture_output() as captured:
        result = f(bad)
    assert result is None
    assert 'Traceback' in captured.stdout
    assert '_repr_pretty_' in captured.stdout
    assert 'given' in captured.stdout
    assert 'argument' in captured.stdout