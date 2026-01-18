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
def test_corrupted_object_raise_exception(self):
    """Corrupted sha1 disk file should raise specific exception."""
    self.store.add_object(testobject)
    self.assertEqual((Blob.type_num, b'yummy data'), self.store.get_raw(testobject.id))
    self.assertTrue(self.store.contains_loose(testobject.id))
    self.assertIsNotNone(self.store._get_loose_object(testobject.id))
    path = self.store._get_shafile_path(testobject.id)
    old_mode = os.stat(path).st_mode
    os.chmod(path, 384)
    with open(path, 'wb') as f:
        f.write(b'')
    os.chmod(path, old_mode)
    expected_error_msg = 'Corrupted empty file detected'
    try:
        self.store.contains_loose(testobject.id)
    except EmptyFileException as e:
        self.assertEqual(str(e), expected_error_msg)
    try:
        self.store._get_loose_object(testobject.id)
    except EmptyFileException as e:
        self.assertEqual(str(e), expected_error_msg)
    self.assertEqual([testobject.id], list(self.store._iter_loose_objects()))