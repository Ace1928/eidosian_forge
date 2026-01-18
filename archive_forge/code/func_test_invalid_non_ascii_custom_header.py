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
def test_invalid_non_ascii_custom_header(self):
    unicode_header = 'x-%s-meta-soufflé:5' % self.provider_custom_meta
    stderr = self.RunGsUtil(['setmeta', '-h', unicode_header, '%s://foo/bar' % self.default_provider], expected_status=1, return_stderr=True)
    self.assertIn('Invalid non-ASCII header', stderr)