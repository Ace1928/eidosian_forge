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
def test_conf_file_re_register_opt(self):
    opt = cfg.StrOpt('foo')
    self.assertTrue(self.conf.register_opt(opt))
    self.assertFalse(self.conf.register_opt(opt))