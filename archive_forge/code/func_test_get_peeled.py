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
def test_get_peeled(self):
    refs = InfoRefsContainer(BytesIO(_TEST_REFS_SERIALIZED))
    self.assertEqual(_TEST_REFS[b'refs/heads/master'], refs.get_peeled(b'refs/heads/master'))