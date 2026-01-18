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
def test_conf_files_mutate_group(self):
    """Test that mutable opts in groups can be reloaded."""
    self.conf.register_cli_opt(cfg.StrOpt('boo', mutable=True), group=self.my_group)
    self._test_conf_files_mutate()
    self.assertTrue(hasattr(self.conf, 'group'))
    self.assertTrue(hasattr(self.conf.group, 'boo'))
    self.assertEqual('new_boo', self.conf.group.boo)