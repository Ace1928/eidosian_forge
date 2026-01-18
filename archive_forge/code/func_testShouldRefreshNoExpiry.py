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
def testShouldRefreshNoExpiry(self):
    """Tests token.ShouldRefresh with no expiry time."""
    mock_datetime = MockDateTime()
    start = datetime.datetime(2011, 3, 1, 11, 25, 13, 300826)
    token = oauth2_client.AccessToken('foo', None, datetime_strategy=mock_datetime)
    mock_datetime.mock_now = start
    self.assertFalse(token.ShouldRefresh())
    mock_datetime.mock_now = start + datetime.timedelta(minutes=472)
    self.assertFalse(token.ShouldRefresh())