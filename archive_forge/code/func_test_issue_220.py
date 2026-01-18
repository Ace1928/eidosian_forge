from __future__ import absolute_import, print_function, unicode_literals
import pytest  # NOQA
import sys
from .roundtrip import (
def test_issue_220(self, tmpdir):
    program_src = '\n        from srsly.ruamel_yaml import YAML\n\n        yaml_str = u"""\\\n        ---\n        foo: ["bar"]\n        """\n\n        yaml = YAML(typ=\'safe\', pure=True)\n        d = yaml.load(yaml_str)\n        print(d)\n        '
    assert save_and_run(dedent(program_src), tmpdir, optimized=True) == 0