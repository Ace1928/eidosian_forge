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
def test__str_override(self):
    self.conf.register_opt(cfg.StrOpt('foo'))
    self.conf.set_override('foo', True)
    self.conf([])
    self.assertEqual('True', self.conf.foo)
    self.conf.clear_override('foo')
    self.assertIsNone(self.conf.foo)