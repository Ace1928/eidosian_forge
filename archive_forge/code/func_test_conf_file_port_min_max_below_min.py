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
def test_conf_file_port_min_max_below_min(self):
    self.conf.register_opt(cfg.PortOpt('foo', min=100, max=500))
    paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = 99\n')])
    self.conf(['--config-file', paths[0]])
    self.assertRaises(cfg.ConfigFileValueError, self.conf._get, 'foo')