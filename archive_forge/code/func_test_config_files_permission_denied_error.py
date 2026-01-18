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
def test_config_files_permission_denied_error(self):
    msg = str(cfg.ConfigFilesPermissionDeniedError(['foo', 'bar']))
    self.assertEqual('Failed to open some config files: foo,bar', msg)