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
def test_conf_file_port_list(self):
    self.conf.register_opt(cfg.ListOpt('foo', item_type=types.Port()))
    paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = 22, 80\n')])
    self.conf(['--config-file', paths[0]])
    self.assertTrue(hasattr(self.conf, 'foo'))
    self.assertEqual([22, 80], self.conf.foo)