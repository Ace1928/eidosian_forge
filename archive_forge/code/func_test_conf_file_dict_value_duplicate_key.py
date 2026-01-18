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
def test_conf_file_dict_value_duplicate_key(self):
    self.conf.register_opt(cfg.DictOpt('foo'))
    paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = key:bar,key:baz\n')])
    self.conf(['--config-file', paths[0]])
    self.assertRaises(cfg.ConfigFileValueError, self.conf._get, 'foo')
    self.assertRaises(ValueError, getattr, self.conf, 'foo')