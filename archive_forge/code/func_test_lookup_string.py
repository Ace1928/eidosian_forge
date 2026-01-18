from math import pi
import pytest
from IPython import get_ipython
from traitlets.config import Config
from IPython.core.formatters import (
from IPython.utils.io import capture_output
def test_lookup_string():
    f = PlainTextFormatter()
    type_str = '%s.%s' % (C.__module__, 'C')
    f.for_type(type_str, foo_printer)
    assert f.lookup(C()) is foo_printer
    assert _mod_name_key(C) not in f.deferred_printers
    assert C in f.type_printers