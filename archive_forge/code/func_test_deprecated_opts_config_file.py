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
def test_deprecated_opts_config_file(self):
    oldopts = [cfg.DeprecatedOpt(self.opt_name)]
    self.conf.register_opt(cfg.BoolOpt('foobar', deprecated_opts=oldopts))
    paths = self.create_tempfiles([('test', '[DEFAULT]\n' + self.cf_name + ' = True\n')])
    self.conf(['--config-file', paths[0]])
    self.assertTrue(self.conf.foobar)