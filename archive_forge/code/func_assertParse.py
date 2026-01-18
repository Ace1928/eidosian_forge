import os
import shutil
import tempfile
from dulwich.tests import TestCase
from ..errors import ObjectFormatException
from ..objects import Tree
from ..repo import MemoryRepo, Repo, parse_graftpoints, serialize_graftpoints
def assertParse(self, expected, graftpoints):
    self.assertEqual(expected, parse_graftpoints(iter(graftpoints)))