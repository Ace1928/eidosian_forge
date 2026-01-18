from math import pi
import pytest
from IPython import get_ipython
from traitlets.config import Config
from IPython.core.formatters import (
from IPython.utils.io import capture_output
def test_bad_repr_traceback():
    f = PlainTextFormatter()
    bad = BadRepr()
    with capture_output() as captured:
        result = f(bad)
    assert result is None
    assert 'Traceback' in captured.stdout
    assert '__repr__' in captured.stdout
    assert 'ValueError' in captured.stdout