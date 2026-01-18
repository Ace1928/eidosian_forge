import pytest
from textwrap import dedent
import platform
import srsly
from .roundtrip import (
def test_duplicate_key_01(self):
    from srsly.ruamel_yaml import version_info
    from srsly.ruamel_yaml.constructor import DuplicateKeyError
    s = dedent('        - &name-name\n          a: 1\n        - &help-name\n          b: 2\n        - <<: *name-name\n          <<: *help-name\n        ')
    if version_info < (0, 15, 1):
        pass
    else:
        with pytest.raises(DuplicateKeyError):
            yaml = YAML(typ='safe')
            yaml.load(s)
        with pytest.raises(DuplicateKeyError):
            yaml = YAML()
            yaml.load(s)