import os
import sys
import tempfile
from io import BytesIO
from typing import ClassVar, Dict
from dulwich import errors
from dulwich.tests import SkipTest, TestCase
from ..file import GitFile
from ..objects import ZERO_SHA
from ..refs import (
from ..repo import Repo
from .utils import open_repo, tear_down_repo
def test_cyrillic(self):
    if sys.platform in ('darwin', 'win32'):
        raise SkipTest("filesystem encoding doesn't support arbitrary bytes")
    name = b'\xcd\xee\xe2\xe0\xff\xe2\xe5\xf2\xea\xe01'
    encoded_ref = b'refs/heads/' + name
    with open(os.path.join(os.fsencode(self._repo.path), encoded_ref), 'w') as f:
        f.write('00' * 20)
    expected_refs = set(_TEST_REFS.keys())
    expected_refs.add(encoded_ref)
    self.assertEqual(expected_refs, set(self._repo.refs.allkeys()))
    self.assertEqual({r[len(b'refs/'):] for r in expected_refs if r.startswith(b'refs/')}, set(self._repo.refs.subkeys(b'refs/')))
    expected_refs.remove(b'refs/heads/loop')
    expected_refs.add(b'HEAD')
    self.assertEqual(expected_refs, set(self._repo.get_refs().keys()))