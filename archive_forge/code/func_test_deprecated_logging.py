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
def test_deprecated_logging(self):
    self.conf.register_opt(cfg.StrOpt('foo', deprecated_name='bar'))
    self.conf.register_group(cfg.OptGroup('other'))
    self.conf.register_opt(cfg.StrOpt('foo', deprecated_name='bar'), group='other')
    if self.deprecated:
        content = 'bar=baz'
    else:
        content = 'foo=baz'
    paths = self.create_tempfiles([('test', '[' + self.group + ']\n' + content + '\n')])
    self.conf(['--config-file', paths[0]])
    if self.group == 'DEFAULT':
        self.assertEqual('baz', self.conf.foo)
        self.assertEqual('baz', self.conf.foo)
    else:
        self.assertEqual('baz', self.conf.other.foo)
        self.assertEqual('baz', self.conf.other.foo)
    if self.deprecated:
        expected = 'Deprecated: ' + cfg._Namespace._deprecated_opt_message % {'dep_option': 'bar', 'dep_group': self.group, 'option': 'foo', 'group': self.group} + '\n'
    else:
        expected = ''
    self.assertEqual(expected, self.log_fixture.output)