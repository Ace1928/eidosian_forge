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
def test_conf_files_reload_error(self):
    self.conf.register_cli_opt(cfg.StrOpt('foo', required=True))
    self.conf.register_cli_opt(cfg.StrOpt('foo1', required=True))
    paths = self.create_tempfiles([('1', '[DEFAULT]\nfoo = test1\nfoo1 = test11\n'), ('2', '[DEFAULT]\nfoo2 = test2\nfoo3 = test22\n')])
    self.conf(['--config-file', paths[0]])
    self.assertTrue(hasattr(self.conf, 'foo'))
    self.assertEqual('test1', self.conf.foo)
    self.assertTrue(hasattr(self.conf, 'foo1'))
    self.assertEqual('test11', self.conf.foo1)
    shutil.copy(paths[1], paths[0])
    self.conf.reload_config_files()
    self.assertTrue(hasattr(self.conf, 'foo'))
    self.assertEqual('test1', self.conf.foo)
    self.assertTrue(hasattr(self.conf, 'foo1'))
    self.assertEqual('test11', self.conf.foo1)