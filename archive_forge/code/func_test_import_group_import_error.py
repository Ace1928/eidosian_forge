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
def test_import_group_import_error(self):
    self.assertRaises(ImportError, cfg.CONF.import_group, 'qux', 'oslo_config.tests.testmods.bazzz_quxxx_opt')