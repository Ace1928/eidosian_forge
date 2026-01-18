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
def test_hooks_see_new_values(self):

    def foo(conf, fresh):
        self.assertEqual('new_foo', conf.foo)
    self.conf.register_cli_opt(cfg.StrOpt('foo', mutable=True))
    self.conf.register_mutate_hook(foo)
    paths = self.create_tempfiles([('1', '[DEFAULT]\nfoo = old_foo\n[group]\nboo = old_boo\n'), ('2', '[DEFAULT]\nfoo = new_foo\n[group]\nboo = new_boo\n')])
    self.conf(['--config-file', paths[0]])
    self.assertEqual('old_foo', self.conf.foo)
    shutil.copy(paths[1], paths[0])
    self.conf.mutate_config_files()
    self.assertEqual('new_foo', self.conf.foo)