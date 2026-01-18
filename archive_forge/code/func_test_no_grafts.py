import os
import shutil
import tempfile
from dulwich.tests import TestCase
from ..errors import ObjectFormatException
from ..objects import Tree
from ..repo import MemoryRepo, Repo, parse_graftpoints, serialize_graftpoints
def test_no_grafts(self):
    r = self.get_repo_with_grafts({})
    shas = [e.commit.id for e in r.get_walker()]
    self.assertEqual(shas, self._shas[::-1])