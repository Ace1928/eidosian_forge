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
def test_conf_files_reload(self):
    self.conf.register_cli_opt(cfg.StrOpt('foo'))
    paths = self.create_tempfiles([('1', '[DEFAULT]\nfoo = baar\n'), ('2', '[DEFAULT]\nfoo = baaar\n')])
    self.conf(['--config-file', paths[0]])
    self.assertTrue(hasattr(self.conf, 'foo'))
    self.assertEqual('baar', self.conf.foo)
    shutil.copy(paths[1], paths[0])
    self.conf.reload_config_files()
    self.assertTrue(hasattr(self.conf, 'foo'))
    self.assertEqual('baaar', self.conf.foo)