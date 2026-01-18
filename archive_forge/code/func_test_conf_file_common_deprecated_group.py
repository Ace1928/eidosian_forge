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
def test_conf_file_common_deprecated_group(self):
    self.conf.register_group(cfg.OptGroup('foo'))
    self.conf.register_group(cfg.OptGroup('bar'))
    oldopts = [cfg.DeprecatedOpt('foo', group='DEFAULT')]
    self.conf.register_opt(cfg.StrOpt('common_opt', deprecated_opts=oldopts), group='bar')
    self.conf.register_opt(cfg.StrOpt('common_opt', deprecated_opts=oldopts), group='foo')
    paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = bla\n')])
    self.conf(['--config-file', paths[0]])
    self.assertEqual('bla', self.conf.foo.common_opt)
    self.assertEqual('bla', self.conf.bar.common_opt)
    paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = bla\n[bar]\ncommon_opt = blabla\n')])
    self.conf(['--config-file', paths[0]])
    self.assertEqual('bla', self.conf.foo.common_opt)
    self.assertEqual('blabla', self.conf.bar.common_opt)
    paths = self.create_tempfiles([('test', '[foo]\ncommon_opt = bla\n[bar]\ncommon_opt = blabla\n')])
    self.conf(['--config-file', paths[0]])
    self.assertEqual('bla', self.conf.foo.common_opt)
    self.assertEqual('blabla', self.conf.bar.common_opt)