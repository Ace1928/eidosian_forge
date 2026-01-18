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
def test_deprecated_opts_not_equal(self):
    d1 = cfg.DeprecatedOpt('oldfoo', group='oldgroup')
    d2 = cfg.DeprecatedOpt('oldfoo2', group='oldgroup')
    self.assertNotEqual(d1, d2)