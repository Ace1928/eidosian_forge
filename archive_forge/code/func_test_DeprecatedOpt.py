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
def test_DeprecatedOpt(self):
    default_deprecated = [cfg.DeprecatedOpt('bar')]
    other_deprecated = [cfg.DeprecatedOpt('baz', group='other')]
    self.conf.register_group(cfg.OptGroup('other'))
    self.conf.register_opt(cfg.StrOpt('foo', deprecated_opts=default_deprecated))
    self.conf.register_opt(cfg.StrOpt('foo', deprecated_opts=other_deprecated), group='other')
    paths = self.create_tempfiles([('test', '[DEFAULT]\n' + 'bar=baz\n' + '[other]\n' + 'baz=baz\n')])
    self.conf(['--config-file', paths[0]])
    self.assertEqual('baz', self.conf.foo)
    self.assertEqual('baz', self.conf.other.foo)
    self.assertIn('Option "bar" from group "DEFAULT"', self.log_fixture.output)
    self.assertIn('Option "baz" from group "other"', self.log_fixture.output)