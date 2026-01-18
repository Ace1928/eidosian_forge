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
def test_find_config_dirs_snap(self):
    config_dirs = ['/var/snap/nova/common/etc/blaa/blaa.conf.d']
    fake_env = {'SNAP': '/snap/nova/current/', 'SNAP_COMMON': '/var/snap/nova/common/'}
    self.useFixture(fixtures.MonkeyPatch('sys.argv', ['foo']))
    self.useFixture(fixtures.MonkeyPatch('os.path.exists', lambda p: p in config_dirs))
    self.useFixture(fixtures.MonkeyPatch('os.environ', fake_env))
    self.assertEqual(cfg.find_config_dirs(project='blaa'), config_dirs)