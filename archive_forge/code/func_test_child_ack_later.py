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
def test_child_ack_later(self):
    gw = self.get_walker([b'a'], {b'a': [b'b'], b'b': [b'c'], b'c': []})
    self.assertEqual(b'a' * 40, next(gw))
    self.assertEqual(b'b' * 40, next(gw))
    gw.ack(b'a' * 40)
    self.assertIs(None, next(gw))