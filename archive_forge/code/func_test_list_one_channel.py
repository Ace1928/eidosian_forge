from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import re
import time
import uuid
import boto
from gslib.cloud_api_delegator import CloudApiDelegator
import gslib.tests.testcase as testcase
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import unittest
from gslib.utils.retry_util import Retry
from six import add_move, MovedModule
from six.moves import mock
@unittest.skipUnless(NOTIFICATION_URL, 'Test requires notification URL configuration.')
def test_list_one_channel(self):
    """Tests listing notification channel on a bucket."""
    return unittest.skip('Functionality has been disabled due to b/132277269')
    bucket_uri = self.CreateBucket()
    self.RunGsUtil(['notification', 'watchbucket', NOTIFICATION_URL, suri(bucket_uri)], return_stderr=False)

    @Retry(AssertionError, tries=3, timeout_secs=5)
    def _ListObjectChangeNotifications():
        stderr = self.RunGsUtil(['notification', 'list', '-o', suri(bucket_uri)], return_stderr=True)
        return stderr
    time.sleep(5)
    stderr = _ListObjectChangeNotifications()
    channel_id = re.findall('Channel identifier: (?P<id>.*)', stderr)
    self.assertEqual(len(channel_id), 1)
    resource_id = re.findall('Resource identifier: (?P<id>.*)', stderr)
    self.assertEqual(len(resource_id), 1)
    push_url = re.findall('Application URL: (?P<id>.*)', stderr)
    self.assertEqual(len(push_url), 1)
    subscriber_email = re.findall('Created by: (?P<id>.*)', stderr)
    self.assertEqual(len(subscriber_email), 1)
    creation_time = re.findall('Creation time: (?P<id>.*)', stderr)
    self.assertEqual(len(creation_time), 1)