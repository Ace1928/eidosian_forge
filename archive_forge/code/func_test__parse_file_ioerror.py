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
def test__parse_file_ioerror(self):
    filename = 'fake'
    namespace = mock.Mock()
    with mock.patch('oslo_config.cfg.ConfigParser.parse') as parse:
        parse.side_effect = IOError(errno.EMFILE, filename, 'Too many open files')
        self.assertRaises(IOError, cfg.ConfigParser._parse_file, filename, namespace)