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
def test_add_pack(self):
    o = DiskObjectStore(self.store_dir)
    self.addCleanup(o.close)
    f, commit, abort = o.add_pack()
    try:
        b = make_object(Blob, data=b'more yummy data')
        write_pack_objects(f.write, [(b, None)])
    except BaseException:
        abort()
        raise
    else:
        commit()