from math import pi
import pytest
from IPython import get_ipython
from traitlets.config import Config
from IPython.core.formatters import (
from IPython.utils.io import capture_output
def test_error_method():
    f = HTMLFormatter()

    class BadHTML(object):

        def _repr_html_(self):
            raise ValueError('Bad HTML')
    bad = BadHTML()
    with capture_output() as captured:
        result = f(bad)
    assert result is None
    assert 'Traceback' in captured.stdout
    assert 'Bad HTML' in captured.stdout
    assert '_repr_html_' in captured.stdout