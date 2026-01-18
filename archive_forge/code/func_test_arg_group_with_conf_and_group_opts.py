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
def test_arg_group_with_conf_and_group_opts(self):
    self.conf.register_cli_opt(cfg.StrOpt('conf'), group='blaa')
    self.conf.register_cli_opt(cfg.StrOpt('group'), group='blaa')
    self.conf(['--blaa-conf', 'foo', '--blaa-group', 'bar'])
    self.assertTrue(hasattr(self.conf, 'blaa'))
    self.assertTrue(hasattr(self.conf.blaa, 'conf'))
    self.assertEqual('foo', self.conf.blaa.conf)
    self.assertTrue(hasattr(self.conf.blaa, 'group'))
    self.assertEqual('bar', self.conf.blaa.group)