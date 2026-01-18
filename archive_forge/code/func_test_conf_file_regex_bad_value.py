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
def test_conf_file_regex_bad_value(self):
    self.conf.register_opt(cfg.StrOpt('foo', regex='bar'))
    paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = other\n')])
    self.conf(['--config-file', paths[0]])
    self.assertRaisesRegex(cfg.ConfigFileValueError, "doesn't match regex", self.conf._get, 'foo')
    self.assertRaisesRegex(ValueError, "doesn't match regex", getattr, self.conf, 'foo')