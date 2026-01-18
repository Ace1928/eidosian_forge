from __future__ import absolute_import, print_function, unicode_literals
import pytest  # NOQA
import sys
from .roundtrip import (
def test_issue_223(self):
    import srsly.ruamel_yaml
    yaml = srsly.ruamel_yaml.YAML(typ='safe')
    yaml.load('phone: 0123456789')