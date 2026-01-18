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
def test_no_default_config_dir(self):
    self.conf(args=[])
    self.assertEqual([], self.conf.config_dir)