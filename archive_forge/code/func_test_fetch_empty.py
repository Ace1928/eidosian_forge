import base64
import os
import shutil
import sys
import tempfile
import warnings
from io import BytesIO
from typing import Dict
from unittest.mock import patch
from urllib.parse import quote as urlquote
from urllib.parse import urlparse
import dulwich
from dulwich import client
from dulwich.tests import TestCase, skipIf
from ..client import (
from ..config import ConfigDict
from ..objects import Commit, Tree
from ..pack import pack_objects_to_data, write_pack_data, write_pack_objects
from ..protocol import TCP_GIT_PORT, Protocol
from ..repo import MemoryRepo, Repo
from .utils import open_repo, setup_warning_catcher, tear_down_repo
def test_fetch_empty(self):
    c = LocalGitClient()
    s = open_repo('a.git')
    self.addCleanup(tear_down_repo, s)
    out = BytesIO()
    walker = {}
    ret = c.fetch_pack(s.path, lambda heads, **kwargs: [], graph_walker=walker, pack_data=out.write)
    self.assertEqual({b'HEAD': b'a90fa2d900a17e99b433217e988c4eb4a2e9a097', b'refs/heads/master': b'a90fa2d900a17e99b433217e988c4eb4a2e9a097', b'refs/tags/mytag': b'28237f4dc30d0d462658d6b937b08a0f0b6ef55a', b'refs/tags/mytag-packed': b'b0931cadc54336e78a1d980420e3268903b57a50'}, ret.refs)
    self.assertEqual({b'HEAD': b'refs/heads/master'}, ret.symrefs)
    self.assertEqual(b'PACK\x00\x00\x00\x02\x00\x00\x00\x00\x02\x9d\x08\x82;\xd8\xa8\xea\xb5\x10\xadj\xc7\\\x82<\xfd>\xd3\x1e', out.getvalue())