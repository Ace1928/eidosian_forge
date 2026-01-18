from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import re
import six
from gslib.commands import defacl
from gslib.cs_api_map import ApiSelector
import gslib.tests.testcase as case
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
from gslib.utils.constants import UTF8
from gslib.utils import shim_util
from six import add_move, MovedModule
from six.moves import mock
def testEmptyDefAcl(self):
    bucket = self.CreateBucket()
    self.RunGsUtil(self._defacl_set_prefix + ['private', suri(bucket)])
    stdout = self.RunGsUtil(self._defacl_get_prefix + [suri(bucket)], return_stdout=True)
    self.assertEqual(stdout.rstrip(), '[]')
    self.RunGsUtil(self._defacl_ch_prefix + ['-u', self.USER_TEST_ADDRESS + ':fc', suri(bucket)])