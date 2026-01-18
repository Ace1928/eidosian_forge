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
def test_get_packed_refs(self):
    self.assertEqual({b'refs/heads/packed': b'42d06bd4b77fed026b154d16493e5deab78f02ec', b'refs/tags/refs-0.1': b'df6800012397fb85c56e7418dd4eb9405dee075c'}, self._refs.get_packed_refs())