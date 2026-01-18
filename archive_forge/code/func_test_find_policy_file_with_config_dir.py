import argparse
import errno
import functools
import io
import logging
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock
import fixtures
from oslotest import base
import testscenarios
from oslo_config import cfg
from oslo_config import types
def test_find_policy_file_with_config_dir(self):
    dir = tempfile.mkdtemp()
    self.tempdirs.append(dir)
    dir2 = tempfile.mkdtemp()
    self.tempdirs.append(dir2)
    path = self.create_tempfiles([(os.path.join(dir, 'policy.json'), '{}')], ext='')[0]
    self.conf(['--config-dir', dir, '--config-dir', dir2])
    self.assertEqual(path, self.conf.find_file('policy.json'))