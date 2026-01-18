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
def test_conf_file_str_value_override_use_deprecated(self):
    """last option should always win, even if last uses deprecated."""
    self.conf.register_cli_opt(cfg.StrOpt('newfoo', deprecated_name='oldfoo'))
    paths = self.create_tempfiles([('0', '[DEFAULT]\nnewfoo = middle\n'), ('1', '[DEFAULT]\noldfoo = last\n')])
    self.conf(['--newfoo', 'first', '--config-file', paths[0], '--config-file', paths[1]])
    self.assertTrue(hasattr(self.conf, 'newfoo'))
    self.assertTrue(hasattr(self.conf, 'oldfoo'))
    self.assertEqual('last', self.conf.newfoo)
    log_out = self.logger.output
    self.assertIn('is deprecated', log_out)
    self.assertIn('Use option "newfoo"', log_out)