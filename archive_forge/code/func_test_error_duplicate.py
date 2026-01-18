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
def test_error_duplicate(self):
    self.conf.register_cli_opt(cfg.StrOpt('foo', help='bar'))
    self.assertRaises(cfg.DuplicateOptError, self.conf.register_cli_opt, cfg.StrOpt('foo'))