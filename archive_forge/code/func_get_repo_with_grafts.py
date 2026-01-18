import os
import shutil
import tempfile
from dulwich.tests import TestCase
from ..errors import ObjectFormatException
from ..objects import Tree
from ..repo import MemoryRepo, Repo, parse_graftpoints, serialize_graftpoints
def get_repo_with_grafts(self, grafts):
    r = self._repo
    r._add_graftpoints(grafts)
    return r