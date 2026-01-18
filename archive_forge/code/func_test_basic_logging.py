import contextlib
import glob
import logging
import os
import re
import subprocess
import sys
import tempfile
import unittest
import warnings
from tornado.escape import utf8
from tornado.log import LogFormatter, define_logging_options, enable_pretty_logging
from tornado.options import OptionParser
from tornado.util import basestring_type
def test_basic_logging(self):
    self.logger.error('foo')
    self.assertEqual(self.get_output(), b'foo')