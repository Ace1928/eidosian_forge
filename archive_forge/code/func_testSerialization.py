from __future__ import absolute_import
import datetime
import logging
import os
import stat
import sys
import unittest
from freezegun import freeze_time
from gcs_oauth2_boto_plugin import oauth2_client
import httplib2
def testSerialization(self):
    """Tests token serialization."""
    expiry = datetime.datetime(2011, 3, 1, 11, 25, 13, 300826)
    token = oauth2_client.AccessToken('foo', expiry, rapt_token=RAPT_TOKEN)
    serialized_token = token.Serialize()
    LOG.debug('testSerialization: serialized_token=%s', serialized_token)
    token2 = oauth2_client.AccessToken.UnSerialize(serialized_token)
    self.assertEqual(token, token2)