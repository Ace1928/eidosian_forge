from __future__ import absolute_import, print_function, unicode_literals
import pytest  # NOQA
import sys
from .roundtrip import (
def test_issue_279(self):
    from srsly.ruamel_yaml import YAML
    from srsly.ruamel_yaml.compat import StringIO
    yaml = YAML()
    yaml.indent(sequence=4, offset=2)
    inp = dedent('        experiments:\n          - datasets:\n        # ATLAS EWK\n              - {dataset: ATLASWZRAP36PB, frac: 1.0}\n              - {dataset: ATLASZHIGHMASS49FB, frac: 1.0}\n        ')
    a = yaml.load(inp)
    buf = StringIO()
    yaml.dump(a, buf)
    print(buf.getvalue())
    assert buf.getvalue() == inp