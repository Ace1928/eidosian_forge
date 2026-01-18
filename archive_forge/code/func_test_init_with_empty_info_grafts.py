import os
import shutil
import tempfile
from dulwich.tests import TestCase
from ..errors import ObjectFormatException
from ..objects import Tree
from ..repo import MemoryRepo, Repo, parse_graftpoints, serialize_graftpoints
def test_init_with_empty_info_grafts(self):
    r = self._repo
    r._put_named_file(os.path.join('info', 'grafts'), b'')
    r = Repo(self._repo_dir)
    self.assertEqual({}, r._graftpoints)