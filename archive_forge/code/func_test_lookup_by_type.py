from math import pi
import pytest
from IPython import get_ipython
from traitlets.config import Config
from IPython.core.formatters import (
from IPython.utils.io import capture_output
def test_lookup_by_type():
    f = PlainTextFormatter()
    f.for_type(C, foo_printer)
    assert f.lookup_by_type(C) is foo_printer
    with pytest.raises(KeyError):
        f.lookup_by_type(A)