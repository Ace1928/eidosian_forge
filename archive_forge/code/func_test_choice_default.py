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
def test_choice_default(self):
    self.conf.register_cli_opt(cfg.PortOpt('port', default=455, choices=[80, 455]))
    self.conf([])
    self.assertEqual(455, self.conf.port)