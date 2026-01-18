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
def test_dict_sub_default_from_default_recurse(self):
    self.conf.register_cli_opt(cfg.StrOpt('foo', default='$foo2'))
    self.conf.register_cli_opt(cfg.StrOpt('foo2', default='floo'))
    self.conf.register_cli_opt(cfg.StrOpt('bar', default='$bar2'))
    self.conf.register_cli_opt(cfg.StrOpt('bar2', default='blaa'))
    self.conf.register_cli_opt(cfg.DictOpt('dt', default={'$foo': '$bar'}))
    self.conf([])
    self.assertEqual('blaa', self.conf.dt['floo'])