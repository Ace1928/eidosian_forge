from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import logging
import unittest
from gslib.cs_api_map import ApiSelector
from gslib.project_id import PopulateProjectId
from gslib.pubsub_api import PubsubApi
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import ObjectToURI as suri
def test_list_new_bucket(self):
    """Tests listing notification configs on a new bucket."""
    if self.test_api == ApiSelector.XML:
        return unittest.skip('Notifications only work with the JSON API.')
    bucket_uri = self.CreateBucket()
    stdout = self.RunGsUtil(['notification', 'list', suri(bucket_uri)], return_stdout=True)
    self.assertFalse(stdout)