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
def test_import_opt_no_such_opt(self):
    self.assertRaises(cfg.NoSuchOptError, cfg.CONF.import_opt, 'blaablaa', 'oslo_config.tests.testmods.blaa_opt')