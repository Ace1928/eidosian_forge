from __future__ import print_function
import sys
import textwrap
import pytest
from pathlib import Path
def test_dump_path(self, tmpdir):
    from srsly.ruamel_yaml import YAML
    fn = Path(str(tmpdir)) / 'test.yaml'
    yaml = YAML()
    data = yaml.map()
    data['a'] = 1
    data['b'] = 2
    yaml.dump(data, fn)
    assert fn.read_text() == 'a: 1\nb: 2\n'