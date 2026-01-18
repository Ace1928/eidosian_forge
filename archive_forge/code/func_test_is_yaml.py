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
def test_is_yaml(self):
    self.assertTrue(fileutils.is_yaml(self.yaml_file))
    self.assertFalse(fileutils.is_yaml(self.json_file))