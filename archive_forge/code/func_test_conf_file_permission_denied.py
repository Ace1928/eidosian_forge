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
@unittest.skipIf(os.getuid() == 0, 'Not supported with the root privileges')
def test_conf_file_permission_denied(self):
    fd, path = tempfile.mkstemp()
    os.chmod(path, 0)
    self.assertRaises(cfg.ConfigFilesPermissionDeniedError, self.conf, ['--config-file', path])
    os.remove(path)