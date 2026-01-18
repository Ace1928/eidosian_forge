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
def test_generic_io_error(self):
    tempdir = tempfile.mkdtemp()
    self.assertRaises(IOError, fileutils.compute_file_checksum, tempdir)