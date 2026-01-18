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
def test_conf_file_raw_value(self):
    self.conf.register_opt(cfg.StrOpt('foo'))
    paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = bar-%08x\n')])
    self.conf(['--config-file', paths[0]])
    self.assertTrue(hasattr(self.conf, 'foo'))
    self.assertEqual('bar-%08x', self.conf.foo)