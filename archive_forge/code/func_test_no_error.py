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
def test_no_error(self):
    tmpfile = tempfile.mktemp()
    open(tmpfile, 'w')
    with fileutils.remove_path_on_error(tmpfile):
        pass
    self.assertTrue(os.path.exists(tmpfile))
    os.unlink(tmpfile)