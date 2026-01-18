from __future__ import print_function
import sys
import textwrap
import pytest
from pathlib import Path
def test_duplicate_keys_00(self):
    from srsly.ruamel_yaml import YAML
    from srsly.ruamel_yaml.constructor import DuplicateKeyError
    yaml = YAML()
    with pytest.raises(DuplicateKeyError):
        yaml.load('{a: 1, a: 2}')