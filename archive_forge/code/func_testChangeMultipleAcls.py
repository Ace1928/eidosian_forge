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
def testChangeMultipleAcls(self):
    """Tests defacl ch with multiple ACL entries."""
    bucket = self.CreateBucket()
    test_regex_group = self._MakeScopeRegex('READER', 'group', self.GROUP_TEST_ADDRESS)
    test_regex_user = self._MakeScopeRegex('OWNER', 'user', self.USER_TEST_ADDRESS)
    json_text = self.RunGsUtil(self._defacl_get_prefix + [suri(bucket)], return_stdout=True)
    self.assertNotRegex(json_text, test_regex_group)
    self.assertNotRegex(json_text, test_regex_user)
    self.RunGsUtil(self._defacl_ch_prefix + ['-g', self.GROUP_TEST_ADDRESS + ':READ', '-u', self.USER_TEST_ADDRESS + ':fc', suri(bucket)])
    json_text = self.RunGsUtil(self._defacl_get_prefix + [suri(bucket)], return_stdout=True)
    self.assertRegex(json_text, test_regex_group)
    self.assertRegex(json_text, test_regex_user)