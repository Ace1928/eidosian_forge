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
def test_file_with_not_default_suffix(self):
    suffix = '.conf'
    res = fileutils.write_to_tempfile(self.content, suffix=suffix)
    self.assertTrue(os.path.exists(res))
    basepath, tmpfile = os.path.split(res)
    self.assertTrue(basepath.startswith(tempfile.gettempdir()))
    self.assertTrue(tmpfile.startswith('tmp'))
    self.assertTrue(tmpfile.endswith('.conf'))
    self.check_file_content(res)