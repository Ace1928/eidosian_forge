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
def testChangeDefaultAclEmpty(self):
    """Tests adding and removing an entry from an empty default object ACL."""
    bucket = self.CreateBucket()
    self.RunGsUtil(self._defacl_set_prefix + ['private', suri(bucket)])
    json_text = self.RunGsUtil(self._defacl_get_prefix + [suri(bucket)], return_stdout=True)
    empty_regex = '\\[\\]\\s*'
    self.assertRegex(json_text, empty_regex)
    group_regex = self._MakeScopeRegex('READER', 'group', self.GROUP_TEST_ADDRESS)
    self.RunGsUtil(self._defacl_ch_prefix + ['-g', self.GROUP_TEST_ADDRESS + ':READ', suri(bucket)])
    json_text2 = self.RunGsUtil(self._defacl_get_prefix + [suri(bucket)], return_stdout=True)
    self.assertRegex(json_text2, group_regex)
    if self.test_api == ApiSelector.JSON:
        return
    self.RunGsUtil(self._defacl_ch_prefix + ['-d', self.GROUP_TEST_ADDRESS, suri(bucket)])
    json_text3 = self.RunGsUtil(self._defacl_get_prefix + [suri(bucket)], return_stdout=True)
    self.assertRegex(json_text3, empty_regex)