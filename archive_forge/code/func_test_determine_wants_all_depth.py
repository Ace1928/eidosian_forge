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
@skipUnless(patch, 'Required mock.patch')
def test_determine_wants_all_depth(self):
    self.store.add_object(testobject)
    refs = {b'refs/heads/foo': testobject.id}
    with patch.object(self.store, '_get_depth', return_value=1) as m:
        self.assertEqual([], self.store.determine_wants_all(refs, depth=0))
        self.assertEqual([testobject.id], self.store.determine_wants_all(refs, depth=DEPTH_INFINITE))
        m.assert_not_called()
        self.assertEqual([], self.store.determine_wants_all(refs, depth=1))
        m.assert_called_with(testobject.id)
        self.assertEqual([testobject.id], self.store.determine_wants_all(refs, depth=2))