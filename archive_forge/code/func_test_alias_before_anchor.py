import pytest
from textwrap import dedent
import platform
import srsly
from .roundtrip import (
@pytest.mark.skipif(platform.python_implementation() == 'Jython', reason='Jython throws RepresenterError')
def test_alias_before_anchor(self):
    from srsly.ruamel_yaml.composer import ComposerError
    with pytest.raises(ComposerError):
        data = load('\n            d: *id002\n            a: &id002\n              b: 1\n              c: 2\n            ')
        data = data