from __future__ import print_function
import sys
import textwrap
import pytest
from pathlib import Path
@pytest.mark.skipif(sys.version_info >= (3, 0), reason='ok on Py3')
def test_duplicate_keys_02(self):
    from srsly.ruamel_yaml import safe_load
    from srsly.ruamel_yaml.constructor import DuplicateKeyError
    with pytest.raises(DuplicateKeyError):
        safe_load('type: Doméstica\ntype: International')