from __future__ import absolute_import, print_function, unicode_literals
import pytest  # NOQA
import sys
from .roundtrip import (
def test_issue_286(self):
    from srsly.ruamel_yaml import YAML
    from srsly.ruamel_yaml.compat import StringIO
    yaml = YAML()
    inp = dedent('        parent_key:\n        - sub_key: sub_value\n\n        # xxx')
    a = yaml.load(inp)
    a['new_key'] = 'new_value'
    buf = StringIO()
    yaml.dump(a, buf)
    assert buf.getvalue().endswith('xxx\nnew_key: new_value\n')