from __future__ import absolute_import
import os
import textwrap
from gslib.commands.rpo import RpoCommand
from gslib.exception import CommandException
from gslib.gcs_json_api import GcsJsonApi
from gslib.storage_url import StorageUrlFromString
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForGS
from gslib.tests.testcase.integration_testcase import SkipForJSON
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.utils import shim_util
from six import add_move, MovedModule
from six.moves import mock
def test_get_with_wildcard(self):
    self.CreateBucket(bucket_name='boo1')
    self.CreateBucket(bucket_name='boo2')
    stdout = self.RunCommand('rpo', ['get', 'gs://boo*'], return_stdout=True)
    actual = '\n'.join(sorted(stdout.strip().split('\n')))
    expected_string = textwrap.dedent('      gs://boo1: None\n      gs://boo2: None')
    self.assertEqual(actual, expected_string)