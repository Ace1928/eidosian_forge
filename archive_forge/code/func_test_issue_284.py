from __future__ import absolute_import, print_function, unicode_literals
import pytest  # NOQA
import sys
from .roundtrip import (
def test_issue_284(self):
    import srsly.ruamel_yaml
    inp = dedent('        plain key: in-line value\n        : # Both empty\n        "quoted key":\n        - entry\n        ')
    yaml = srsly.ruamel_yaml.YAML(typ='rt')
    yaml.version = (1, 2)
    d = yaml.load(inp)
    assert d[None] is None
    yaml = srsly.ruamel_yaml.YAML(typ='rt')
    yaml.version = (1, 1)
    with pytest.raises(srsly.ruamel_yaml.parser.ParserError, match='expected <block end>'):
        d = yaml.load(inp)