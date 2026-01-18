from __future__ import absolute_import, print_function, unicode_literals
import pytest  # NOQA
import sys
from .roundtrip import (
def test_issue_234(self):
    from srsly.ruamel_yaml import YAML
    inp = dedent('        - key: key1\n          ctx: [one, two]\n          help: one\n          cmd: >\n            foo bar\n            foo bar\n        ')
    yaml = YAML(typ='safe', pure=True)
    data = yaml.load(inp)
    fold = data[0]['cmd']
    print(repr(fold))
    assert '\x07' not in fold