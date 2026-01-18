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
def test_log_file_with_timed_rotating(self):
    tmpdir = tempfile.mkdtemp()
    try:
        self.options.log_file_prefix = tmpdir + '/test_log'
        self.options.log_rotate_mode = 'time'
        enable_pretty_logging(options=self.options, logger=self.logger)
        self.logger.error('hello')
        self.logger.handlers[0].flush()
        filenames = glob.glob(tmpdir + '/test_log*')
        self.assertEqual(1, len(filenames))
        with open(filenames[0], encoding='utf-8') as f:
            self.assertRegex(f.read(), '^\\[E [^]]*\\] hello$')
    finally:
        for handler in self.logger.handlers:
            handler.flush()
            handler.close()
        for filename in glob.glob(tmpdir + '/test_log*'):
            os.unlink(filename)
        os.rmdir(tmpdir)