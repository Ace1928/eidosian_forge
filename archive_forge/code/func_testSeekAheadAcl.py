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
def testSeekAheadAcl(self):
    """Tests seek-ahead iterator with ACL sub-commands."""
    object_uri = self.CreateObject(contents=b'foo')
    current_acl = self.RunGsUtil(['acl', 'get', suri(object_uri)], return_stdout=True)
    current_acl_file = self.CreateTempFile(contents=current_acl.encode(UTF8))
    with SetBotoConfigForTest([('GSUtil', 'task_estimation_threshold', '1'), ('GSUtil', 'task_estimation_force', 'True')]):
        stderr = self.RunGsUtil(['-m', 'acl', 'ch', '-u', 'AllUsers:R', suri(object_uri)], return_stderr=True)
        self.assertIn('Estimated work for this command: objects: 1\n', stderr)
        stderr = self.RunGsUtil(['-m', 'acl', 'set', current_acl_file, suri(object_uri)], return_stderr=True)
        self.assertIn('Estimated work for this command: objects: 1\n', stderr)
    with SetBotoConfigForTest([('GSUtil', 'task_estimation_threshold', '0'), ('GSUtil', 'task_estimation_force', 'True')]):
        stderr = self.RunGsUtil(['-m', 'acl', 'ch', '-u', 'AllUsers:R', suri(object_uri)], return_stderr=True)
        self.assertNotIn('Estimated work', stderr)