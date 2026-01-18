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
def test_deprecated_for_removal_with_group(self):
    self.conf.register_group(cfg.OptGroup('other'))
    self.conf.register_opt(cfg.StrOpt('foo', deprecated_for_removal=True), group='other')
    self.conf.register_opt(cfg.StrOpt('bar', deprecated_for_removal=True), group='other')
    paths = self.create_tempfiles([('test', '[other]\n' + 'foo=bar\n')])
    self.conf(['--config-file', paths[0]])
    self.assertEqual('bar', self.conf.other.foo)
    self.assertEqual('bar', self.conf.other.foo)
    self.assertIsNone(self.conf.other.bar)
    expected = 'Option "foo" from group "other" is deprecated for removal.  Its value may be silently ignored in the future.\n'
    self.assertIn(expected, self.log_fixture.output)