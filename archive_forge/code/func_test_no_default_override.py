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
def test_no_default_override(self):
    self.conf.register_opt(cfg.StrOpt('foo'))
    self.conf([])
    self.assertIsNone(self.conf.foo)
    self.conf.set_default('foo', 'bar')
    self.assertEqual('bar', self.conf.foo)
    self.conf.clear_default('foo')
    self.assertIsNone(self.conf.foo)