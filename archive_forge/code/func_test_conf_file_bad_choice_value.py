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
def test_conf_file_bad_choice_value(self):
    self.conf.register_opt(cfg.PortOpt('port', choices=[80, 8080]))
    paths = self.create_tempfiles([('test', '[DEFAULT]\nport = 8181\n')])
    self.conf(['--config-file', paths[0]])
    self.assertRaises(cfg.ConfigFileValueError, self.conf._get, 'port')
    self.assertRaises(ValueError, getattr, self.conf, 'port')