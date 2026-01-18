from __future__ import print_function
import sys
import textwrap
import pytest
from pathlib import Path
def test_multi_document_load(self, tmpdir):
    """this went wrong on 3.7 because of StopIteration, PR 37 and Issue 211"""
    from srsly.ruamel_yaml import YAML
    fn = Path(str(tmpdir)) / 'test.yaml'
    fn.write_text(textwrap.dedent(u'            ---\n            - a\n            ---\n            - b\n            ...\n            '))
    yaml = YAML()
    assert list(yaml.load_all(fn)) == [['a'], ['b']]