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
def test_get_for_multiple_bucket_calls_api(self):
    bucket_uri1 = self.CreateBucket(bucket_name='rpofoo')
    bucket_uri2 = self.CreateBucket(bucket_name='rpobar')
    stdout = self.RunCommand('rpo', ['get', suri(bucket_uri1), suri(bucket_uri2)], return_stdout=True)
    expected_string = textwrap.dedent('      gs://rpofoo: None\n      gs://rpobar: None\n      ')
    self.assertEqual(expected_string, stdout)