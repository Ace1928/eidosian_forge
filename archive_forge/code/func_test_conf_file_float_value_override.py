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
def test_conf_file_float_value_override(self):
    self.conf.register_cli_opt(cfg.FloatOpt('foo'))
    paths = self.create_tempfiles([('1', '[DEFAULT]\nfoo = 6.6\n'), ('2', '[DEFAULT]\nfoo = 6.66\n')])
    self.conf(['--foo', '6', '--config-file', paths[0], '--config-file', paths[1]])
    self.assertTrue(hasattr(self.conf, 'foo'))
    self.assertEqual(6.66, self.conf.foo)