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
def testMultiVersionSupport(self):
    """Tests changing ACLs on multiple object versions."""
    bucket = self.CreateVersionedBucket()
    object_name = self.MakeTempName('obj')
    obj1_uri = self.CreateObject(bucket_uri=bucket, object_name=object_name, contents=b'One thing')
    self.CreateObject(bucket_uri=bucket, object_name=object_name, contents=b'Another thing', gs_idempotent_generation=urigen(obj1_uri))
    lines = self.AssertNObjectsInBucket(bucket, 2, versioned=True)
    obj_v1, obj_v2 = (lines[0], lines[1])
    test_regex = self._MakeScopeRegex('READER', 'group', self.GROUP_TEST_ADDRESS)
    json_text = self.RunGsUtil(self._get_acl_prefix + [obj_v1], return_stdout=True)
    self.assertNotRegex(json_text, test_regex)
    self.RunGsUtil(self._ch_acl_prefix + ['-g', self.GROUP_TEST_ADDRESS + ':READ', obj_v1])
    json_text = self.RunGsUtil(self._get_acl_prefix + [obj_v1], return_stdout=True)
    self.assertRegex(json_text, test_regex)
    json_text = self.RunGsUtil(self._get_acl_prefix + [obj_v2], return_stdout=True)
    self.assertNotRegex(json_text, test_regex)