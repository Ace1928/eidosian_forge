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
def test_check_refname(self):
    self._refs._check_refname(b'HEAD')
    self._refs._check_refname(b'refs/stash')
    self._refs._check_refname(b'refs/heads/foo')
    self.assertRaises(errors.RefFormatError, self._refs._check_refname, b'refs')
    self.assertRaises(errors.RefFormatError, self._refs._check_refname, b'notrefs/foo')