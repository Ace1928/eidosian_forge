import os
import shutil
import stat
import sys
import tempfile
from contextlib import closing
from io import BytesIO
from unittest import skipUnless
from dulwich.tests import TestCase
from ..errors import NotTreeError
from ..index import commit_tree
from ..object_store import (
from ..objects import (
from ..pack import REF_DELTA, write_pack_objects
from ..protocol import DEPTH_INFINITE
from .utils import build_pack, make_object, make_tag
def test_add_objects(self):
    data = [(testobject, 'mypath')]
    self.store.add_objects(data)
    self.assertEqual({testobject.id}, set(self.store))
    self.assertIn(testobject.id, self.store)
    r = self.store[testobject.id]
    self.assertEqual(r, testobject)