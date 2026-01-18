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
def testUniqeId(self):
    cred_id = self.client.CacheKey()
    self.assertEqual('0720afed6871f12761fbea3271f451e6ba184bf5', cred_id)