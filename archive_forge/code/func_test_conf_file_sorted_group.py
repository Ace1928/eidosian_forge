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
def test_conf_file_sorted_group(self):
    for i in range(10):
        group = cfg.OptGroup('group%s' % i, 'options')
        self.conf.register_group(group)
        self.conf.register_cli_opt(cfg.StrOpt('opt1'), group=group)
    paths = self.create_tempfiles([('test', '[group1]\nopt1 = foo\n[group2]\nopt2 = bar\n')])
    self.conf(['--config-file', paths[0]])
    self.assertEqual('foo', self.conf.group1.opt1)