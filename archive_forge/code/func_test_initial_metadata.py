from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import re
from unittest import mock
import six
from gslib.commands import setmeta
from gslib.cs_api_map import ApiSelector
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
from gslib.utils.retry_util import Retry
from gslib.utils import shim_util
def test_initial_metadata(self):
    """Tests copying file to an object with metadata."""
    objuri = suri(self.CreateObject(contents=b'foo'))
    inpath = self.CreateTempFile()
    ct = 'image/gif'
    self.RunGsUtil(['-h', 'x-%s-meta-xyz:abc' % self.provider_custom_meta, '-h', 'Content-Type:%s' % ct, 'cp', inpath, objuri])

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check1():
        stdout = self.RunGsUtil(['ls', '-L', objuri], return_stdout=True)
        self.assertRegex(stdout, 'Content-Type:\\s+%s' % ct)
        self.assertRegex(stdout, 'xyz:\\s+abc')
    _Check1()