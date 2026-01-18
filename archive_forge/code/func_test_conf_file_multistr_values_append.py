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
def test_conf_file_multistr_values_append(self):
    self.conf.register_cli_opt(cfg.MultiStrOpt('foo'))
    paths = self.create_tempfiles([('1', '[DEFAULT]\nfoo = bar1\n'), ('2', '[DEFAULT]\nfoo = bar2\nfoo = bar3\n')])
    self.conf(['--foo', 'bar0', '--config-file', paths[0], '--config-file', paths[1]])
    self.assertTrue(hasattr(self.conf, 'foo'))
    self.assertEqual(['bar0', 'bar1', 'bar2', 'bar3'], self.conf.foo)