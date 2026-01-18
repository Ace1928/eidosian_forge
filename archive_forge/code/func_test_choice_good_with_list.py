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
def test_choice_good_with_list(self):
    self.conf.register_cli_opt(cfg.PortOpt('port', choices=[80, 8080]))
    self.conf(['--port', '80'])
    self.assertEqual(80, self.conf.port)