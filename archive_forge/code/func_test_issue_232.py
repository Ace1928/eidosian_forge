from __future__ import absolute_import, print_function, unicode_literals
import pytest  # NOQA
import sys
from .roundtrip import (
def test_issue_232(self):
    import srsly.ruamel_yaml
    import srsly.ruamel_yaml as yaml
    with pytest.raises(srsly.ruamel_yaml.parser.ParserError):
        yaml.safe_load(']')
    with pytest.raises(srsly.ruamel_yaml.parser.ParserError):
        yaml.safe_load('{]')