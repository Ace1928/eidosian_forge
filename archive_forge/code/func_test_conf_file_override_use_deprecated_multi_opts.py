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
def test_conf_file_override_use_deprecated_multi_opts(self):
    self.conf.register_group(cfg.OptGroup('blaa'))
    oldopts = [cfg.DeprecatedOpt('oldfoo', group='oldgroup'), cfg.DeprecatedOpt('oldfoo2', group='oldgroup2')]
    self.conf.register_cli_opt(cfg.StrOpt('foo', deprecated_opts=oldopts), group='blaa')
    paths = self.create_tempfiles([('test', '[oldgroup2]\noldfoo2 = bar\n')])
    self.conf(['--config-file', paths[0]])
    self.assertEqual('bar', self.conf.blaa.foo)