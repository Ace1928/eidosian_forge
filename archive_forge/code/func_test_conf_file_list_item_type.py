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
def test_conf_file_list_item_type(self):
    self.conf.register_cli_opt(cfg.ListOpt('foo', item_type=types.Integer()))
    paths = self.create_tempfiles([('1', '[DEFAULT]\nfoo = 1,2\n')])
    self.conf(['--config-file', paths[0]])
    self.assertTrue(hasattr(self.conf, 'foo'))
    self.assertEqual([1, 2], self.conf.foo)