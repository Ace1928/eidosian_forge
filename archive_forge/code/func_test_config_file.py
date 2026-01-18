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
def test_config_file(self):
    paths = self.create_tempfiles([('1', '[DEFAULT]'), ('2', '[DEFAULT]')])
    self.conf(['--config-file', paths[0], '--config-file', paths[1]])
    self.assertEqual(paths, self.conf.config_file)