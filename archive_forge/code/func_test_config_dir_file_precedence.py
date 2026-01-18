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
def test_config_dir_file_precedence(self):
    snafu_group = cfg.OptGroup('snafu')
    self.conf.register_group(snafu_group)
    self.conf.register_cli_opt(cfg.StrOpt('foo'))
    self.conf.register_cli_opt(cfg.StrOpt('bell'), group=snafu_group)
    dir = tempfile.mkdtemp()
    self.tempdirs.append(dir)
    paths = self.create_tempfiles([(os.path.join(dir, '00-test'), '[DEFAULT]\nfoo = bar-00\n'), ('01-test', '[snafu]\nbell = whistle-01\n[DEFAULT]\nfoo = bar-01\n'), ('03-test', '[snafu]\nbell = whistle-03\n[DEFAULT]\nfoo = bar-03\n'), (os.path.join(dir, '02-test'), '[DEFAULT]\nfoo = bar-02\n')])
    self.conf(['--foo', 'bar', '--config-file', paths[1], '--config-dir', os.path.dirname(paths[0]), '--config-file', paths[2]])
    self.assertTrue(hasattr(self.conf, 'foo'))
    self.assertEqual('bar-03', self.conf.foo)
    self.assertTrue(hasattr(self.conf, 'snafu'))
    self.assertTrue(hasattr(self.conf.snafu, 'bell'))
    self.assertEqual('whistle-03', self.conf.snafu.bell)