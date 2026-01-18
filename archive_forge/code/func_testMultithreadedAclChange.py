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
def testMultithreadedAclChange(self, count=10):
    """Tests multi-threaded acl changing on several objects."""
    objects = []
    for i in range(count):
        objects.append(self.CreateObject(bucket_uri=self.sample_uri, contents='something {0}'.format(i).encode('ascii')))
    self.AssertNObjectsInBucket(self.sample_uri, count)
    test_regex = self._MakeScopeRegex('READER', 'group', self.GROUP_TEST_ADDRESS)
    json_texts = []
    for obj in objects:
        json_texts.append(self.RunGsUtil(self._get_acl_prefix + [suri(obj)], return_stdout=True))
    for json_text in json_texts:
        self.assertNotRegex(json_text, test_regex)
    uris = [suri(obj) for obj in objects]
    self.RunGsUtil(['-m', '-DD'] + self._ch_acl_prefix + ['-g', self.GROUP_TEST_ADDRESS + ':READ'] + uris)
    json_texts = []
    for obj in objects:
        json_texts.append(self.RunGsUtil(self._get_acl_prefix + [suri(obj)], return_stdout=True))
    for json_text in json_texts:
        self.assertRegex(json_text, test_regex)