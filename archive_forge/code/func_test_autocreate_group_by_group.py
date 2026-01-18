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
def test_autocreate_group_by_group(self):
    group = cfg.OptGroup(name='blaa', title='Blaa options')
    self.conf.register_cli_opt(cfg.StrOpt('foo'), group=group)
    self.conf(['--blaa-foo', 'bar'])
    self.assertTrue(hasattr(self.conf, 'blaa'))
    self.assertTrue(hasattr(self.conf.blaa, 'foo'))
    self.assertEqual('bar', self.conf.blaa.foo)