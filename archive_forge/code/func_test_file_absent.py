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
def test_file_absent(self):
    tmpfile = tempfile.mktemp()
    fileutils.delete_if_exists(tmpfile)
    self.assertFalse(os.path.exists(tmpfile))