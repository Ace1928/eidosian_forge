from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import re
from gslib.commands import acl
from gslib.command import CreateOrGetGsutilLogger
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.storage_url import StorageUrlFromString
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForGS
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import GenerationFromURI as urigen
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
from gslib.utils import acl_helper
from gslib.utils.constants import UTF8
from gslib.utils.retry_util import Retry
from gslib.utils.translation_helper import AclTranslation
from gslib.utils import shim_util
from six import add_move, MovedModule
from six.moves import mock
def testRecursiveChangeAcl(self):
    """Tests recursively changing ACLs on nested objects."""
    obj = self.CreateObject(bucket_uri=self.sample_uri, object_name='foo/bar', contents=b'something')
    self.AssertNObjectsInBucket(self.sample_uri, 1)
    test_regex = self._MakeScopeRegex('READER', 'group', self.GROUP_TEST_ADDRESS)
    json_text = self.RunGsUtil(self._get_acl_prefix + [suri(obj)], return_stdout=True)
    self.assertNotRegex(json_text, test_regex)

    @Retry(AssertionError, tries=5, timeout_secs=1)
    def _AddAcl():
        self.RunGsUtil(self._ch_acl_prefix + ['-R', '-g', self.GROUP_TEST_ADDRESS + ':READ', suri(obj)[:-3]])
        json_text = self.RunGsUtil(self._get_acl_prefix + [suri(obj)], return_stdout=True)
        self.assertRegex(json_text, test_regex)
    _AddAcl()

    @Retry(AssertionError, tries=5, timeout_secs=1)
    def _DeleteAcl():
        delete_grant = self.GROUP_TEST_ADDRESS.upper()
        self.RunGsUtil(self._ch_acl_prefix + ['-d', delete_grant, suri(obj)])
        json_text = self.RunGsUtil(self._get_acl_prefix + [suri(obj)], return_stdout=True)
        self.assertNotRegex(json_text, test_regex)
    _DeleteAcl()