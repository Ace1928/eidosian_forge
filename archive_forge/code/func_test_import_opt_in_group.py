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
def test_import_opt_in_group(self):
    self.assertFalse(hasattr(cfg.CONF, 'bar'))
    cfg.CONF.import_opt('foo', 'oslo_config.tests.testmods.bar_foo_opt', group='bar')
    self.assertTrue(hasattr(cfg.CONF, 'bar'))
    self.assertTrue(hasattr(cfg.CONF.bar, 'foo'))