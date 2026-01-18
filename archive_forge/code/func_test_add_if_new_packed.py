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
def test_add_if_new_packed(self):
    self.assertFalse(self._refs.add_if_new(b'refs/tags/refs-0.1', b'9' * 40))
    self.assertEqual(b'df6800012397fb85c56e7418dd4eb9405dee075c', self._refs[b'refs/tags/refs-0.1'])