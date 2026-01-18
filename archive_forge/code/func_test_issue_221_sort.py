from __future__ import absolute_import, print_function, unicode_literals
import pytest  # NOQA
import sys
from .roundtrip import (
def test_issue_221_sort(self):
    from srsly.ruamel_yaml import YAML
    from srsly.ruamel_yaml.compat import StringIO
    yaml = YAML()
    inp = dedent('        - d\n        - a  # 1\n        - c  # 3\n        - e  # 5\n        - b  # 2\n        ')
    a = yaml.load(dedent(inp))
    a.sort()
    buf = StringIO()
    yaml.dump(a, buf)
    exp = dedent('        - a  # 1\n        - b  # 2\n        - c  # 3\n        - d\n        - e  # 5\n        ')
    assert buf.getvalue() == exp