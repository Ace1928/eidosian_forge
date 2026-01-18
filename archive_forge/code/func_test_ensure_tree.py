import errno
import hashlib
import json
import os
import shutil
import stat
import tempfile
import time
from unittest import mock
import uuid
import yaml
from oslotest import base as test_base
from oslo_utils import fileutils
def test_ensure_tree(self):
    tmpdir = tempfile.mkdtemp()
    try:
        testdir = '%s/foo/bar/baz' % (tmpdir,)
        fileutils.ensure_tree(testdir, TEST_PERMISSIONS)
        self.assertTrue(os.path.isdir(testdir))
        self.assertEqual(os.stat(testdir).st_mode, TEST_PERMISSIONS | stat.S_IFDIR)
    finally:
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)