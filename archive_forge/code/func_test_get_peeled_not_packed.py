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
def test_get_peeled_not_packed(self):
    self.assertEqual(None, self._refs.get_peeled(b'refs/tags/refs-0.2'))
    self.assertEqual(b'3ec9c43c84ff242e3ef4a9fc5bc111fd780a76a8', self._refs[b'refs/tags/refs-0.2'])
    self.assertEqual(self._refs[b'refs/heads/packed'], self._refs.get_peeled(b'refs/heads/packed'))
    self.assertEqual(b'42d06bd4b77fed026b154d16493e5deab78f02ec', self._refs.get_peeled(b'refs/tags/refs-0.1'))