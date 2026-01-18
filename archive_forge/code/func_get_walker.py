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
def get_walker(self, heads, parent_map):
    new_parent_map = {k * 40: [p * 40 for p in ps] for k, ps in parent_map.items()}
    return ObjectStoreGraphWalker([x * 40 for x in heads], new_parent_map.__getitem__)