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
def test_default_config_file_priority(self):
    self.conf.register_cli_opt(cfg.StrOpt('foo'))
    paths = self.create_tempfiles([('def', '[DEFAULT]\nfoo = bar\n')])
    self.conf(args=['--foo=blaa'], default_config_files=paths)
    self.assertEqual(paths, self.conf.config_file)
    self.assertEqual('blaa', self.conf.foo)