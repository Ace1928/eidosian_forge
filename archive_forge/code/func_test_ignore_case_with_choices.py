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
def test_ignore_case_with_choices(self):
    self.conf.register_cli_opt(cfg.StrOpt('foo', ignore_case=True, choices=['bar1', 'bar2', 'BAR3']))
    self.conf(['--foo', 'bAr1'])
    self.assertEqual('bAr1', self.conf.foo)
    self.conf(['--foo', 'BaR2'])
    self.assertEqual('BaR2', self.conf.foo)
    self.conf(['--foo', 'baR3'])
    self.assertEqual('baR3', self.conf.foo)