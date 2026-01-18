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
def test_change_default_many(self):
    opts = [cfg.StrOpt('foo', default='foo'), cfg.StrOpt('foo2', default='foo2')]
    self.conf.register_opts(opts)
    cfg.set_defaults(opts, foo='bar', foo2='bar2')
    self.conf([])
    self.assertEqual('bar', self.conf.foo)
    self.assertEqual('bar2', self.conf.foo2)