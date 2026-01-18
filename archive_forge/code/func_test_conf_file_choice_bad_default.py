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
def test_conf_file_choice_bad_default(self):
    self.assertRaises(cfg.DefaultValueError, cfg.PortOpt, 'port', choices=[80, 8080], default=8181)