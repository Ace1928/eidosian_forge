from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import socket
import sys
import six
import boto
from gslib.commands.perfdiag import _GenerateFileData
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import RUN_S3_TESTS
from gslib.tests.util import unittest
from gslib.utils.system_util import IS_WINDOWS
from six import add_move, MovedModule
from six.moves import mock
@SkipForXML('No compressed transport encoding support for the XML API.')
def test_gzip_write_throughput_single_process_single_thread(self):
    stderr_default, _ = self._run_throughput_test('wthru', 1, 1, compression_ratio=50)
    self.assertIn('Gzip compression ratio: 50', stderr_default)
    self.assertIn('Gzip transport encoding writes: True', stderr_default)
    stderr_default, _ = self._run_throughput_test('wthru_file', 1, 1, compression_ratio=50)
    self.assertIn('Gzip compression ratio: 50', stderr_default)
    self.assertIn('Gzip transport encoding writes: True', stderr_default)