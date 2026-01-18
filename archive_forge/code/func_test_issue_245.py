from __future__ import absolute_import, print_function, unicode_literals
import pytest  # NOQA
import sys
from .roundtrip import (
def test_issue_245(self):
    from srsly.ruamel_yaml import YAML
    inp = '\n        d: yes\n        '
    for typ in ['safepure', 'rt', 'safe']:
        if typ.endswith('pure'):
            pure = True
            typ = typ[:-4]
        else:
            pure = None
        yaml = YAML(typ=typ, pure=pure)
        yaml.version = (1, 1)
        d = yaml.load(inp)
        print(typ, yaml.parser, yaml.resolver)
        assert d['d'] is True