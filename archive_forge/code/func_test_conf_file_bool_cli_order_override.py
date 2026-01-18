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
def test_conf_file_bool_cli_order_override(self):
    self.conf.register_cli_opt(cfg.BoolOpt('foo'))
    paths = self.create_tempfiles([('1', '[DEFAULT]\nfoo = false\n')])
    self.conf(['--config-file', paths[0], '--foo'])
    self.assertTrue(hasattr(self.conf, 'foo'))
    self.assertTrue(self.conf.foo)