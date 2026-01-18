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
def test_ignore_case_with_regex(self):
    self.conf.register_cli_opt(cfg.StrOpt('foo', ignore_case=True, regex='fOO|bar'))
    self.conf(['--foo', 'foo'])
    self.assertEqual('foo', self.conf.foo)
    self.conf(['--foo', 'Bar'])
    self.assertEqual('Bar', self.conf.foo)
    self.conf(['--foo', 'FOObar'])
    self.assertEqual('FOObar', self.conf.foo)