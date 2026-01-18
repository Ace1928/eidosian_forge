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
def test_conf_files_mutate_foo(self):
    """Test that a mutable opt can be reloaded."""
    self.conf.register_cli_opt(cfg.StrOpt('foo', mutable=True))
    self._test_conf_files_mutate()
    self.assertTrue(hasattr(self.conf, 'foo'))
    self.assertEqual('new_foo', self.conf.foo)