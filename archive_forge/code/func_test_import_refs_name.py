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
def test_import_refs_name(self):
    self._refs[b'refs/remotes/origin/other'] = b'48d01bd4b77fed026b154d16493e5deab78f02ec'
    self._refs.import_refs(b'refs/remotes/origin', {b'master': b'42d06bd4b77fed026b154d16493e5deab78f02ec'})
    self.assertEqual(b'42d06bd4b77fed026b154d16493e5deab78f02ec', self._refs[b'refs/remotes/origin/master'])
    self.assertEqual(b'48d01bd4b77fed026b154d16493e5deab78f02ec', self._refs[b'refs/remotes/origin/other'])