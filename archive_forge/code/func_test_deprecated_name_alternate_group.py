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
def test_deprecated_name_alternate_group(self):
    self.conf.register_opt(cfg.StrOpt('foobar', deprecated_name=self.opt_name, deprecated_group='testing'), group=cfg.OptGroup('testing'))
    self.assertTrue(hasattr(self.conf.testing, 'foobar'))
    self.assertTrue(hasattr(self.conf.testing, self.opt_dest))
    self.assertFalse(hasattr(self.conf.testing, self.broken_opt_dest))
    self.assertIn('foobar', self.conf.testing)
    self.assertNotIn(self.opt_dest, self.conf.testing)
    self.assertNotIn(self.broken_opt_dest, self.conf.testing)