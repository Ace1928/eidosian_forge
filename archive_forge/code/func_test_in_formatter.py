from math import pi
import pytest
from IPython import get_ipython
from traitlets.config import Config
from IPython.core.formatters import (
from IPython.utils.io import capture_output
def test_in_formatter():
    f = PlainTextFormatter()
    f.for_type(C, foo_printer)
    type_str = '%s.%s' % (C.__module__, 'C')
    assert C in f
    assert type_str in f