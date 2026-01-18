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
def test_conf_file_multiple_opts(self):
    self.conf.register_opts([cfg.StrOpt('foo'), cfg.StrOpt('bar')])
    paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = bar\nbar = foo\n')])
    self.conf(['--config-file', paths[0]])
    self.assertTrue(hasattr(self.conf, 'foo'))
    self.assertEqual('bar', self.conf.foo)
    self.assertTrue(hasattr(self.conf, 'bar'))
    self.assertEqual('foo', self.conf.bar)