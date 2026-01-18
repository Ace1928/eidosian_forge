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
def test_list_one_entry(self):
    """Tests notification config listing with one entry."""
    if self.test_api == ApiSelector.XML:
        return unittest.skip('Notifications only work with the JSON API.')
    bucket_uri = self.CreateBucket()
    bucket_name = bucket_uri.bucket_name
    topic_name = self._RegisterDefaultTopicCreation(bucket_uri.bucket_name)
    self.RunGsUtil(['notification', 'create', '-f', 'json', '-e', 'OBJECT_FINALIZE', '-e', 'OBJECT_DELETE', '-m', 'someKey:someValue', '-p', 'somePrefix', suri(bucket_uri)], return_stderr=True)
    stdout = self.RunGsUtil(['notification', 'list', suri(bucket_uri)], return_stdout=True)
    if self._use_gcloud_storage:
        trailing_space = '\n'
    else:
        trailing_space = ''
    self.assertEqual(stdout, "projects/_/buckets/{bucket_name}/notificationConfigs/1\n\tCloud Pub/Sub topic: {topic_name}\n\tCustom attributes:\n\t\tsomeKey: someValue\n\tFilters:\n\t\tEvent Types: OBJECT_FINALIZE, OBJECT_DELETE\n\t\tObject name prefix: 'somePrefix'\n{trailing_space}".format(bucket_name=bucket_name, topic_name=topic_name, trailing_space=trailing_space))