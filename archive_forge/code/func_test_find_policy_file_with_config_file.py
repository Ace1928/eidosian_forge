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
def test_find_policy_file_with_config_file(self):
    dir = tempfile.mkdtemp()
    self.tempdirs.append(dir)
    paths = self.create_tempfiles([(os.path.join(dir, 'test.conf'), '[DEFAULT]'), (os.path.join(dir, 'policy.json'), '{}')], ext='')
    self.conf(['--config-file', paths[0]])
    self.assertEqual(paths[1], self.conf.find_file('policy.json'))