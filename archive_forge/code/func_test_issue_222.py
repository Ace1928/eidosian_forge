from __future__ import absolute_import, print_function, unicode_literals
import pytest  # NOQA
import sys
from .roundtrip import (
def test_issue_222(self):
    import srsly.ruamel_yaml
    from srsly.ruamel_yaml.compat import StringIO
    buf = StringIO()
    srsly.ruamel_yaml.safe_dump(['012923'], buf)
    assert buf.getvalue() == "['012923']\n"